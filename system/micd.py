#!/usr/bin/env python3
import numpy as np
from functools import cache
import threading
import time

from cereal import messaging
from openpilot.common.realtime import Ratekeeper
from openpilot.common.retry import retry
from openpilot.common.swaglog import cloudlog

RATE = 10
FFT_SAMPLES = 1600  # 100ms @16k
REFERENCE_SPL = 2e-5  # newtons/m^2
SAMPLE_RATE = 16000
SAMPLE_BUFFER = 800  # 50ms
CANDIDATE_SR = [16000, 48000, 44100]  # 回退采样率


@cache
def get_a_weighting_filter():
  freqs = np.fft.fftfreq(FFT_SAMPLES, d=1 / SAMPLE_RATE)
  A = 12194 ** 2 * freqs ** 4 / (
      (freqs ** 2 + 20.6 ** 2) *
      (freqs ** 2 + 12194 ** 2) *
      np.sqrt((freqs ** 2 + 107.7 ** 2) * (freqs ** 2 + 737.9 ** 2))
  )
  return A / np.max(A)


def calculate_spl(measurements):
  sound_pressure = np.sqrt(np.mean(measurements ** 2))
  if sound_pressure > 0:
    sound_pressure_level = 20 * np.log10(sound_pressure / REFERENCE_SPL)
  else:
    sound_pressure_level = 0
  return sound_pressure, sound_pressure_level


def apply_a_weighting(measurements: np.ndarray) -> np.ndarray:
  measurements_windowed = measurements * np.hanning(len(measurements))
  return np.abs(np.fft.ifft(np.fft.fft(measurements_windowed) * get_a_weighting_filter()))


class DummyStream:
  def __enter__(self): return self
  def __exit__(self, exc_type, exc, tb): pass
  def start(self): pass
  def read(self, frames):
    return (np.zeros((frames, 1), dtype=np.float32), None)


class Mic:
  def __init__(self):
    self.rk = Ratekeeper(RATE)
    self.pm = messaging.PubMaster(['soundPressure', 'rawAudioData'])

    self.measurements = np.empty(0)
    self.sound_pressure = 0
    self.sound_pressure_weighted = 0
    self.sound_pressure_level_weighted = 0

    self.lock = threading.Lock()

  def _list_input_devices(self, sd):
    try:
      devs = sd.query_devices()
      return [i for i, d in enumerate(devs) if d.get('max_input_channels', 0) > 0]
    except Exception as e:
      cloudlog.event("micd: query_devices failed", error=str(e))
      return []

  def _choose_device(self, sd):
    ids = self._list_input_devices(sd)
    if not ids:
      return None
    try:
      default_in = sd.default.device[0]
    except Exception:
      default_in = None
    if default_in in ids:
      return default_in
    return ids[0]

  @retry(3)
  def get_stream(self, sd):
    if sd is None:
      cloudlog.event("micd: sounddevice not available, using DummyStream")
      return DummyStream()

    dev = self._choose_device(sd)
    if dev is None:
      cloudlog.event("micd: no input device found, using DummyStream")
      return DummyStream()

    for sr in CANDIDATE_SR:
      try:
        stream = sd.InputStream(
          device=dev, channels=1, dtype='float32',
          samplerate=sr, blocksize=SAMPLE_BUFFER, latency='low'
        )
        cloudlog.event("micd: opened InputStream", device_index=dev, samplerate=sr)
        return stream
      except Exception as e:
        cloudlog.event("micd: open stream failed, trying fallback",
                       device_index=dev, samplerate=sr, error=str(e))
        time.sleep(0.05)

    cloudlog.event("micd: all attempts failed, using DummyStream")
    return DummyStream()

  def update(self):
    with self.lock:
      sound_pressure = self.sound_pressure
      sound_pressure_weighted = self.sound_pressure_weighted
      sound_pressure_level_weighted = self.sound_pressure_level_weighted

    dat = messaging.new_message('soundPressure')
    dat.soundPressure.soundPressure = float(sound_pressure)
    dat.soundPressure.soundPressureWeighted = float(sound_pressure_weighted)
    dat.soundPressure.soundPressureLevelWeighted = float(sound_pressure_level_weighted)
    self.pm.send('soundPressure', dat)

  def micd_thread(self):
    try:
      import sounddevice as sd
    except Exception as e:
      sd = None
      cloudlog.event("micd: import sounddevice failed", error=str(e))

    with self.get_stream(sd) as stream:
      try:
        stream.start()
      except Exception:
        pass

      while True:
        try:
          data, _ = stream.read(SAMPLE_BUFFER)
          x = np.asarray(data, dtype=np.float32).reshape(-1)

          self.measurements = np.concatenate([self.measurements, x])
          if len(self.measurements) >= FFT_SAMPLES:
            meas = self.measurements[:FFT_SAMPLES]
            self.measurements = self.measurements[FFT_SAMPLES:]

            sp, _ = calculate_spl(meas)
            weighted = apply_a_weighting(meas)
            sp_w, spl_w = calculate_spl(weighted)

            with self.lock:
              self.sound_pressure = sp
              self.sound_pressure_weighted = sp_w
              self.sound_pressure_level_weighted = spl_w

            self.update()
        except Exception as e:
          cloudlog.event("micd: read error", error=str(e))
          time.sleep(0.05)

        self.rk.keep_time()


def main():
  mic = Mic()
  mic.micd_thread()


if __name__ == "__main__":
  main()

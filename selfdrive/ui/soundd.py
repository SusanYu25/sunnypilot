import math
import numpy as np
import time
import wave
import os

from cereal import car, messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import Ratekeeper
from openpilot.common.retry import retry
from openpilot.common.swaglog import cloudlog

from openpilot.system import micd
from openpilot.selfdrive.ui.sunnypilot.quiet_mode import QuietMode

SAMPLE_RATE = 48000
SAMPLE_BUFFER = 4096  # (approx 100ms)
MAX_VOLUME = 1.0
MIN_VOLUME = 0.1
SELFDRIVE_STATE_TIMEOUT = 5  # 5 seconds
FILTER_DT = 1. / (micd.SAMPLE_RATE / micd.FFT_SAMPLES)

AMBIENT_DB = 30  # DB where MIN_VOLUME is applied
DB_SCALE = 30   # AMBIENT_DB + DB_SCALE is where MAX_VOLUME is applied

AudibleAlert = car.CarControl.HUDControl.AudibleAlert


sound_list: dict[int, tuple[str, int | None, float]] = {
  AudibleAlert.engage: ("engage.wav", 1, MAX_VOLUME),
  AudibleAlert.disengage: ("disengage.wav", 1, MAX_VOLUME),
  AudibleAlert.refuse: ("refuse.wav", 1, MAX_VOLUME),

  AudibleAlert.prompt: ("prompt.wav", 1, MAX_VOLUME),
  AudibleAlert.promptRepeat: ("prompt.wav", None, MAX_VOLUME),
  AudibleAlert.promptDistracted: ("prompt_distracted.wav", None, MAX_VOLUME),

  AudibleAlert.warningSoft: ("warning_soft.wav", None, MAX_VOLUME),
  AudibleAlert.warningImmediate: ("warning_immediate.wav", None, MAX_VOLUME),
}


def check_selfdrive_timeout_alert(sm):
  ss_missing = time.monotonic() - sm.recv_time['selfdriveState']
  if ss_missing > SELFDRIVE_STATE_TIMEOUT:
    if sm['selfdriveState'].enabled and (ss_missing - SELFDRIVE_STATE_TIMEOUT) < 10:
      return True
  return False


class DummyStream:
  """Context manager fallback when audio device can't be opened."""
  def __init__(self):
    self.active = False
    self.device = None
  def __enter__(self):
    return self
  def __exit__(self, exc_type, exc, tb):
    return False


class Soundd(QuietMode):
  def __init__(self):
    super().__init__()
    self.load_sounds()
    self.current_alert = AudibleAlert.none
    self.current_volume = MIN_VOLUME
    self.current_sound_frame = 0
    self.selfdrive_timeout_alert = False
    self.spl_filter_weighted = FirstOrderFilter(0, 2.5, FILTER_DT, initialized=False)

  def load_sounds(self):
    self.loaded_sounds: dict[int, np.ndarray] = {}
    for sound in sound_list:
      filename, play_count, volume = sound_list[sound]
      try:
        with wave.open(BASEDIR + "/selfdrive/assets/sounds/" + filename, 'r') as wavefile:
          assert wavefile.getnchannels() == 1
          assert wavefile.getsampwidth() == 2
          assert wavefile.getframerate() == SAMPLE_RATE
          length = wavefile.getnframes()
          self.loaded_sounds[sound] = np.frombuffer(
            wavefile.readframes(length), dtype=np.int16
          ).astype(np.float32) / (2**16/2)
      except Exception as e:
        cloudlog.error(f"Failed to load sound {filename}: {e}")

  def get_sound_data(self, frames):
    ret = np.zeros(frames, dtype=np.float32)
    if self.should_play_sound(self.current_alert) and self.current_alert in self.loaded_sounds:
      num_loops = sound_list[self.current_alert][1]
      sound_data = self.loaded_sounds[self.current_alert]
      written_frames = 0
      current_sound_frame = self.current_sound_frame % len(sound_data)
      loops = self.current_sound_frame // len(sound_data)
      while written_frames < frames and (num_loops is None or loops < num_loops):
        available_frames = sound_data.shape[0] - current_sound_frame
        frames_to_write = min(available_frames, frames - written_frames)
        ret[written_frames:written_frames+frames_to_write] = sound_data[current_sound_frame:current_sound_frame+frames_to_write]
        written_frames += frames_to_write
        self.current_sound_frame += frames_to_write
    return ret * self.current_volume

  def callback(self, data_out: np.ndarray, frames: int, time, status) -> None:
    if status:
      cloudlog.warning(f"soundd stream over/underflow: {status}")
    try:
      data_out[:frames, 0] = self.get_sound_data(frames)
    except Exception as e:
      cloudlog.warning(f"soundd callback error: {e}")

  def update_alert(self, new_alert):
    current_alert_played_once = self.current_alert == AudibleAlert.none or \
                                (self.current_alert in self.loaded_sounds and self.current_sound_frame > len(self.loaded_sounds[self.current_alert]))
    if self.current_alert != new_alert and (new_alert != AudibleAlert.none or current_alert_played_once):
      self.current_alert = new_alert
      self.current_sound_frame = 0

  def get_audible_alert(self, sm):
    if sm.updated['selfdriveState']:
      new_alert = sm['selfdriveState'].alertSound.raw
      self.update_alert(new_alert)
    elif check_selfdrive_timeout_alert(sm):
      self.update_alert(AudibleAlert.warningImmediate)
      self.selfdrive_timeout_alert = True
    elif self.selfdrive_timeout_alert:
      self.update_alert(AudibleAlert.none)
      self.selfdrive_timeout_alert = False

  def calculate_volume(self, weighted_db):
    volume = ((weighted_db - AMBIENT_DB) / DB_SCALE) * (MAX_VOLUME - MIN_VOLUME) + MIN_VOLUME
    return math.pow(10, (np.clip(volume, MIN_VOLUME, MAX_VOLUME) - 1))

  def _choose_output_device(self, sd):
    """Try to find a suitable output device. Allow override via env SOUNDDEVICE_OUTPUT."""
    try:
      env = os.environ.get("SOUNDDEVICE_OUTPUT", None)
      if env is not None:
        try:
          idx = int(env)
          dev = sd.query_devices(idx)
          cloudlog.info(f"Using SOUNDDEVICE_OUTPUT index {idx}: {dev['name']}")
          return idx
        except Exception:
          ds = sd.query_devices()
          for i, d in enumerate(ds):
            if env.lower() in d['name'].lower() and d['max_output_channels'] > 0:
              cloudlog.info(f"Using SOUNDDEVICE_OUTPUT name match {d['name']} (index {i})")
              return i
      devices = sd.query_devices()
      for i, d in enumerate(devices):
        if d['max_output_channels'] > 0:
          cloudlog.info(f"Auto-selected audio output device: {d['name']} (index {i})")
          return i
    except Exception as e:
      cloudlog.warning(f"Device query failed: {e}")
    return None

  @retry(attempts=2, delay=1)
  def get_stream(self, sd):
    """
    Attempt to create an OutputStream. On failure return DummyStream instead of raising.
    """
    try:
      dev_idx = self._choose_output_device(sd)
      if dev_idx is not None:
        try:
          stream = sd.OutputStream(device=dev_idx, channels=1, samplerate=SAMPLE_RATE,
                                   callback=self.callback, blocksize=SAMPLE_BUFFER)
          return stream
        except Exception as e:
          cloudlog.warning(f"Failed to open OutputStream on device {dev_idx}: {e}")
      try:
        stream = sd.OutputStream(channels=1, samplerate=SAMPLE_RATE,
                                 callback=self.callback, blocksize=SAMPLE_BUFFER)
        return stream
      except Exception as e:
        cloudlog.warning(f"Failed to open default OutputStream: {e}")
    except Exception as e:
      cloudlog.warning(f"get_stream unexpected error: {e}")

    cloudlog.error("No usable audio device found, falling back to DummyStream")
    return DummyStream()

  def soundd_thread(self):
    import sounddevice as sd
    sm = messaging.SubMaster(['selfdriveState', 'soundPressure'])
    rk = Ratekeeper(20)

    stream = None
    try:
      stream = self.get_stream(sd)
      if isinstance(stream, DummyStream):
        cloudlog.info("soundd running in silent (dummy) mode")
        while True:
          sm.update(0)
          self.load_param()
          if sm.updated['soundPressure'] and self.current_alert == AudibleAlert.none:
            try:
              self.spl_filter_weighted.update(sm["soundPressure"].soundPressureWeightedDb)
              self.current_volume = self.calculate_volume(float(self.spl_filter_weighted.x))
            except Exception:
              pass
          self.get_audible_alert(sm)
          rk.keep_time()
      else:
        with stream as real_stream:
          cloudlog.info(f"soundd stream started: {getattr(real_stream, 'device', 'default')}")
          while True:
            sm.update(0)
            self.load_param()
            if sm.updated['soundPressure'] and self.current_alert == AudibleAlert.none:
              try:
                self.spl_filter_weighted.update(sm["soundPressure"].soundPressureWeightedDb)
                self.current_volume = self.calculate_volume(float(self.spl_filter_weighted.x))
              except Exception:
                pass
            self.get_audible_alert(sm)
            if not getattr(real_stream, "active", True):
              cloudlog.warning("soundd stream became inactive, entering fallback loop")
              break
            rk.keep_time()
        cloudlog.info("soundd real_stream ended, entering silent fallback")
        while True:
          sm.update(0)
          self.load_param()
          self.get_audible_alert(sm)
          rk.keep_time()

    except Exception as e:
      cloudlog.exception(f"soundd top-level exception, entering silent fallback: {e}")
      while True:
        sm.update(0)
        self.load_param()
        self.get_audible_alert(sm)
        rk.keep_time()


def main():
  s = Soundd()
  s.soundd_thread()


if __name__ == "__main__":
  main()

import pygame.midi
import time
import threading
import random #randomising velocity cause why not

class MidiSoundPlayer:
    def __init__(self, instrument=0, sustain_time=2.0):
        pygame.midi.init()
        self.player = pygame.midi.Output(0)
        self.instrument = instrument
        self.player.set_instrument(self.instrument)

        self.muted = False
        self.volume = 1.0
        self.sustain_time = sustain_time

        self.note_cooldowns = {}  # note: last_played_time
        self.cooldown_time = 2.5  # 2.5 sec cooldown per note (tweakable)

        self.last_note = None
        self.last_note_time = 0
        self.playing_chords = set()

    def set_instrument(self, instrument_id: int):
        self.instrument = instrument_id
        self.player.set_instrument(instrument_id)

    def set_volume(self, volume: float):
        self.volume = max(0.0, min(volume, 1.0))  # Clamp to [0.0, 1.0]

    def toggle_mute(self):
        self.muted = not self.muted

    def play_note(self, note: int, duration: float = None):
        if self.muted:
            return

        duration = duration or self.sustain_time
        now = time.time()
        if note in self.note_cooldowns and (now - self.note_cooldowns[note]) < self.cooldown_time:
            return  # Skip due to cooldown

        # Stop previous note if still ringing and cooldown not over
        if self.last_note and (now - self.last_note_time) < duration:
            self.player.note_off(self.last_note, 127)

        velocity = int(self.volume * 127)
        velocity += random.randint(-3, 3)  # Small random variation
        velocity = max(0, min(127, velocity))  # Clamp between 0–127
        self.player.note_on(note, velocity)
        
        self.note_cooldowns[note] = now
        self.last_note = note
        self.last_note_time = now

        # Threaded stop (non-blocking)
        threading.Thread(target=self._stop_note_after_delay, args=(note, duration), daemon=True).start()

    def play_chord(self, notes: list[int], duration: float = None):
        if self.muted:
            return

        duration = duration or self.sustain_time
        import random

        velocity = int(self.volume * 127)
        velocity += random.randint(-3, 3)  # Small random variation
        velocity = max(0, min(127, velocity))  # Clamp between 0–127

        for note in notes:
            if note not in self.playing_chords:
                self.player.note_on(note, velocity)
                self.playing_chords.add(note)

        threading.Thread(target=self._stop_chord_after_delay, args=(notes, duration), daemon=True).start()

    def _stop_note_after_delay(self, note: int, delay: float):
        time.sleep(delay)
        self.player.note_off(note, 127)
        if self.last_note == note:
            self.last_note = None

    def _stop_chord_after_delay(self, notes: list[int], delay: float):
        time.sleep(delay)
        for note in notes:
            self.player.note_off(note, 127)
            self.playing_chords.discard(note)

    def stop(self):
        # Stop solo note
        if self.last_note:
            self.player.note_off(self.last_note, 127)
            self.last_note = None

        # Stop all chords
        for note in list(self.playing_chords):
            self.player.note_off(note, 127)
        self.playing_chords.clear()

        self.player.close()
        pygame.midi.quit()

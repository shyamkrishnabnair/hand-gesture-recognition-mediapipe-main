# Timing & Synchronization Analysis: Audio vs Notation Panel

## Overview
This document analyzes the timing, delays, cooldowns, and conditions for audio playback and notation panel symbol appending.

---

## 1. Audio Playback (MidiSoundPlayer)

### When Audio is Fired
**Location:** `main.py` line 326
```python
player.play_note(note, duration=10)
```

### Conditions for Audio to Play
**Location:** `main.py` lines 317-320
- ✅ `total_finger_count in note_mapping` (valid gesture)
- ✅ `finger_count_changed or note_cooldown_passed`
  - `finger_count_changed` = `total_finger_count != last_finger_count`
  - `note_cooldown_passed` = `current_time - last_note_time > note_cooldown`

### Cooldowns for Audio
1. **Main Cooldown:** `note_cooldown = 0.5` seconds (line 52)
   - Global cooldown between ANY notes
   - Updated: `last_note_time = current_time` (line 352)

2. **Per-Note Cooldown:** `cooldown_time = 2.5` seconds (midisoundplayer.py line 18)
   - Per-note cooldown in `MidiSoundPlayer.play_note()`
   - Prevents same note from playing within 2.5s
   - ⚠️ **POTENTIAL CONFLICT:** This 2.5s cooldown might override the 0.5s main cooldown

### Audio Execution Flow
1. **Immediate:** `self.player.note_on(note, velocity)` - plays instantly (midisoundplayer.py line 51)
2. **Duration:** Note plays for `duration=10` seconds (from main.py line 326)
3. **Stop:** Threaded stop after delay (midisoundplayer.py line 58)
   - Uses `threading.Thread` with `time.sleep(delay)`
   - Then calls `self.player.note_off(note, 127)`

### Timing Characteristics
- **Latency:** ~0ms (immediate MIDI note_on)
- **Duration:** 10 seconds (as specified in main.py)
- **Frame Rate:** update_frame() runs every ~10ms (line 397: `video_label.after(10, update_frame)`)
- **Max Notes/Second:** 2 notes/second (due to 0.5s cooldown)

---

## 2. Notation Panel Symbol Appending

### When Symbols are Appended
**Location:** `main.py` line 331
```python
notation_panel.add_gesture(total_finger_count)
```

### Conditions for Symbol to Append
**Location:** `main.py` lines 328-332
- ✅ Must be inside the audio trigger condition (line 319)
- ✅ `finger_count_changed or notation_cooldown_passed`
  - `finger_count_changed` = `total_finger_count != last_finger_count`
  - `notation_cooldown_passed` = `current_time - last_notation_time > note_cooldown`

### Cooldowns for Notation
- **Cooldown:** `note_cooldown = 0.5` seconds (same as audio)
- **Updated:** `last_notation_time = current_time` (line 332)
- **Max Symbols/Second:** 2 symbols/second (due to 0.5s cooldown)

### Symbol Execution Flow
1. **Immediate:** `self.canvas.create_text()` - symbol appears instantly (notation_panel.py line 71)
2. **Position:** Symbol appears at right edge: `x = self.canvas.winfo_width() - 20`
3. **Scrolling:** Symbol scrolls left continuously
   - Scroll speed: `2` pixels per frame (notation_panel.py line 4)
   - Frame interval: `50ms` (notation_panel.py line 88: `self.canvas.after(50, self.scroll)`)
   - **Effective scroll speed:** 2px / 50ms = **40 pixels/second**

### Timing Characteristics
- **Latency:** ~0ms (immediate canvas text creation)
- **Scroll Speed:** 40 pixels/second
- **Frame Rate:** Scroll updates every 50ms
- **Max Symbols/Second:** 2 symbols/second (due to 0.5s cooldown)

---

## 3. Synchronization Analysis

### Current Synchronization
✅ **GOOD:** Both audio and notation are triggered in the same code block
✅ **GOOD:** Both use the same cooldown value (0.5s)
✅ **GOOD:** Both check `finger_count_changed` condition

### Potential Issues

#### Issue 1: Conditional Mismatch
- **Audio triggers when:** `finger_count_changed OR note_cooldown_passed`
- **Notation triggers when:** `finger_count_changed OR notation_cooldown_passed`
- ⚠️ **Problem:** If `finger_count_changed=True` but `notation_cooldown_passed=False`, audio plays but notation might not append
- ⚠️ **Problem:** If `note_cooldown_passed=True` but `notation_cooldown_passed=False`, audio plays but notation doesn't append

**Current Code Behavior:**
- If finger count changes → Both fire ✅
- If only note cooldown passes → Audio fires, notation might not fire ❌
- If only notation cooldown passes → Neither fires (because outer condition requires note_cooldown_passed) ❌

#### Issue 2: MidiSoundPlayer Internal Cooldown
- **Main cooldown:** 0.5s (allows 2 notes/second)
- **MidiSoundPlayer cooldown:** 2.5s per note (allows 0.4 notes/second for same note)
- ⚠️ **Conflict:** If same note is played, MidiSoundPlayer's 2.5s cooldown might prevent playback even if main cooldown (0.5s) allows it

#### Issue 3: Timing Precision
- **update_frame()** runs every ~10ms (100 FPS)
- **Notation scroll** updates every 50ms (20 FPS)
- ⚠️ **Mismatch:** Symbol appears immediately but scrolls at different rate than frame updates

---

## 4. Human Time vs Panel Speed

### Notation Panel Scroll Speed
- **Scroll Speed:** 40 pixels/second
- **Canvas Width:** ~900 pixels (default)
- **Time to cross screen:** 900px / 40px/s = **22.5 seconds**

### Audio Duration
- **Note Duration:** 10 seconds
- **Cooldown:** 0.5 seconds
- **Max Notes/Second:** 2 notes/second

### Comparison
- **Audio plays for:** 10 seconds
- **Symbol visible for:** ~22.5 seconds (until it scrolls off screen)
- **Symbol appears:** Immediately when gesture detected
- **Audio starts:** Immediately when gesture detected

✅ **GOOD:** Symbol stays visible longer than audio plays, so user can see what was played

---

## 5. Recommendations

### Fix 1: Synchronize Conditions
Make notation panel use the same condition as audio:
```python
# Current (line 330):
if finger_count_changed or notation_cooldown_passed:

# Should be:
if finger_count_changed or note_cooldown_passed:
```
This ensures notation always appears when audio plays.

### Fix 2: Remove MidiSoundPlayer Cooldown Conflict
Either:
- Remove the 2.5s per-note cooldown in MidiSoundPlayer, OR
- Make it respect the main cooldown (0.5s) instead

### Fix 3: Align Scroll Speed with Frame Rate
Consider making scroll updates match frame rate (every 10ms instead of 50ms) for smoother animation.

---

## Summary Table

| Aspect | Audio | Notation Panel |
|--------|-------|----------------|
| **Trigger Condition** | `finger_count_changed OR note_cooldown_passed` | `finger_count_changed OR notation_cooldown_passed` |
| **Cooldown** | 0.5s (main) + 2.5s (per-note) | 0.5s |
| **Latency** | ~0ms | ~0ms |
| **Max Rate** | 2/second (main) or 0.4/second (per-note) | 2/second |
| **Duration** | 10 seconds | ~22.5 seconds (visible) |
| **Update Rate** | Every ~10ms (frame rate) | Every 50ms (scroll) |


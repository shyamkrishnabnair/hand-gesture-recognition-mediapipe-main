import customtkinter as ctk
import time

class NotationPanel:
    def __init__(self, parent, width=900, height=150, scroll_speed=2.5):
        self.canvas = ctk.CTkCanvas(
            parent, width=width, height=height, bg="black", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.scroll_speed = scroll_speed
        self.events = []   # stores canvas text IDs

        # Symbols for each finger count
        self.symbols = {
            1: "●", 2: "▲", 3: "■", 4: "★", 5: "✦",
            6: "✧", 7: "♫", 8: "♩", 9: "♪", 10: "♬"
        }


        # Draw staff lines (5 horizontal lines)
        self.staff_y_positions = []
        margin_top = 40
        spacing = 15  # space between lines
        for i in range(5):
            y = margin_top + i * spacing
            self.staff_y_positions.append(y)

        # Store line IDs for dynamic resizing
        self.staff_lines = []

        # Draw initial staff lines
        self.draw_staff_lines()

        # Bind resize event to redraw lines
        self.canvas.bind("<Configure>", self.draw_staff_lines)

        # Map finger counts to staff positions
        self.note_positions = {
            1: self.staff_y_positions[4],  # bottom line
            2: self.staff_y_positions[3],
            3: self.staff_y_positions[2],  # middle line
            4: self.staff_y_positions[1],
            5: self.staff_y_positions[0],  # top line
            6: self.staff_y_positions[2] - 20,
            7: self.staff_y_positions[2] + 20,
            8: self.staff_y_positions[0] - 20,
            9: self.staff_y_positions[4] + 20,
            10: self.staff_y_positions[2],
        }

        # Start scrolling loop
        self.scroll()

    def draw_staff_lines(self, event=None):
        """Draw or redraw staff lines to match canvas width."""
        # Clear old lines
        for line_id in self.staff_lines:
            self.canvas.delete(line_id)
        self.staff_lines.clear()

        canvas_width = self.canvas.winfo_width() or 1
        for y in self.staff_y_positions:
            line_id = self.canvas.create_line(0, y, canvas_width, y, fill="white")
            self.staff_lines.append(line_id)

    def add_gesture(self, gesture_id: int):
        """Draw a new note for a gesture."""
        symbol = self.symbols.get(gesture_id, "?")
        x = self.canvas.winfo_width() - 20
        y = self.note_positions.get(gesture_id, self.staff_y_positions[2])

        text_id = self.canvas.create_text(
            x, y, text=symbol, fill="white", font=("Segoe UI", 20, "bold")
        )
        self.events.append(text_id)

    def scroll(self):
        """Continuously move notes left."""
        for item in self.events:
            self.canvas.move(item, -self.scroll_speed, 0)

        # Remove off-screen notes
        for item in self.events[:]:
            coords = self.canvas.coords(item)
            if coords and coords[0] < -20:
                self.canvas.delete(item)
                self.events.remove(item)

        self.canvas.after(50, self.scroll)

    def get_note_symbol(self, gesture_id):
        return self.symbols.get(gesture_id, "?")

    def get_note_y(self, gesture_id):
        return self.note_positions.get(
            gesture_id,
            self.staff_y_positions[2]  # fallback: middle line
        )
    def add_timed_event(self, event, timeline_start_time):
        """
        event = { "gesture": int, "time": float, "instrument": int }
        timeline_start_time = float (when playback/recording started)
        """

        now = time.time() - timeline_start_time
        age = now - event["time"]    # seconds since event happened

        if age < 0:
            return  # not played yet (future event)

        # convert scroll speed (px per 50ms) into px/sec
        scroll_speed_px_per_sec = self.scroll_speed * 20

        # live X position
        x = self.canvas.winfo_width() - age * scroll_speed_px_per_sec

        symbol = self.get_note_symbol(event["gesture"])
        y = self.get_note_y(event["gesture"])

        text_id = self.canvas.create_text(
            x, y, text=symbol, fill="white",
            font=("Segoe UI", 20, "bold")
        )

        # store tuple: canvas id + event reference
        self.events.append((text_id, event))


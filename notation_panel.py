import customtkinter as ctk

class NotationPanel:
    def __init__(self, parent, width=900, height=120):
        self.canvas = ctk.CTkCanvas(
            parent, width=width, height=height, bg="black", highlightthickness=0
        )
        self.canvas.pack(fill="x", expand=True)

        self.events = []   # stores canvas text IDs
        self.symbols = {
            1: "●", 2: "▲", 3: "■", 4: "★", 5: "✦",
            6: "✧", 7: "♫", 8: "♩", 9: "♪", 10: "♬"
        }

        # Start auto-scrolling
        self.scroll()

    def add_gesture(self, gesture_id):
        """Add a symbol for the given gesture"""
        symbol = self.symbols.get(gesture_id, "?")
        x = self.canvas.winfo_width() - 20
        y = 60
        text_id = self.canvas.create_text(
            x, y, text=symbol, fill="white", font=("Segoe UI", 20, "bold")
        )
        self.events.append(text_id)

    def scroll(self):
        """Scroll symbols left and cleanup"""
        for item in self.events:
            self.canvas.move(item, -2, 0)

        for item in self.events[:]:
            x, y = self.canvas.coords(item)
            if x < -20:
                self.canvas.delete(item)
                self.events.remove(item)

        self.canvas.after(50, self.scroll)

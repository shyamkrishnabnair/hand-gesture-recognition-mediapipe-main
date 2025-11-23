import logging
# --- Custom logging handler for CTkTextbox ---
class CTkTextboxHandler(logging.Handler):
    def __init__(self, textbox_widget):
        super().__init__()
        self.textbox = textbox_widget
        self.textbox.tag_config("INFO", foreground="white")
        self.textbox.tag_config("WARNING", foreground="yellow")
        self.textbox.tag_config("ERROR", foreground="red")
        self.textbox.tag_config("CRITICAL", foreground="red") 
        self.textbox.tag_config("DEBUG", foreground="gray")

    def emit(self, record):
        msg = self.format(record)
        self.textbox.after(0, self._insert_log, msg + "\n", record.levelname)

    def _insert_log(self, msg, levelname):
        max_lines = 100 
        if int(self.textbox.index('end-1c').split('.')[0]) > max_lines:
            self.textbox.delete(1.0, 2.0) 

        # Insert the new message at the end, applying a tag for coloring
        self.textbox.insert("end", msg, levelname.upper())
        # Automatically scroll to the end to show the latest messages
        self.textbox.see("end")
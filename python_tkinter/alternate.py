import tkinter as tk

class TwoWindowsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Two Windows App")

        self.window1_btn = tk.Button(root, text="Open Window 1", command=self.open_window1)
        self.window1_btn.pack(pady=10)

        self.window2_btn = tk.Button(root, text="Open Window 2", command=self.open_window2)
        self.window2_btn.pack(pady=10)

        self.current_window = None

    def open_window1(self):
        if self.current_window:
            self.current_window.destroy()

        window1 = tk.Toplevel(self.root)
        window1.title("Window 1")

        label = tk.Label(window1, text="This is Window 1")
        label.pack(padx=20, pady=20)

        switch_button = tk.Button(window1, text="Switch to Window 2", command=self.open_window2)
        switch_button.pack(pady=10)

        self.current_window = window1

    def open_window2(self):
        if self.current_window:
            self.current_window.destroy()

        window2 = tk.Toplevel(self.root)
        window2.title("Window 2")

        label = tk.Label(window2, text="This is Window 2")
        label.pack(padx=20, pady=20)

        switch_button = tk.Button(window2, text="Switch to Window 1", command=self.open_window1)
        switch_button.pack(pady=10)

        self.current_window = window2

if __name__ == "__main__":
    root = tk.Tk()
    app = TwoWindowsApp(root)
    root.mainloop()

import tkinter as tk
from tkinter import ttk

def button_func():
    print('A button was pressed')


#create window
window = tk.Tk()
window.title('Window and widget')
window.geometry('800x500')


# widgets
# master == parent
text = tk.Text(master = window)
text.pack()


#ttk widgets
label = ttk.Label(master = window,text='this is a test')
label.pack()

#ttk entry 
entry = ttk.Entry(master = window)
entry.pack()


#ttk button 
button = ttk.Button(master = window,text = 'button',command = button_func)
button.pack()



# run
window.mainloop()

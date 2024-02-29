import tkinter as tk
from tkinter import ttk

def button_func():
    print('A button was pressed')

def button_func2():
    print('HEllo')



#create window
window = tk.Tk()
window.title('Window and widget')
window.geometry('800x500')


# widgets
# master == parent
text = tk.Text(master = window)
text.pack()


#ttk label
label = ttk.Label(master = window,text='this is a test')
label.pack()



#ttk entry 
entry = ttk.Entry(master = window)
entry.pack()


label2 = ttk.Label(master = window,text = 'My label')
label2.pack()



#ttk button 
# command only wants the NAME of the function, wihtout "()" 
button = ttk.Button(master = window,text = 'button', command = button_func)
button.pack()

butto2 = ttk.Button(master = window,text = 'button2', command = lambda: print('Hello'))
butto2.pack()


# run
window.mainloop()

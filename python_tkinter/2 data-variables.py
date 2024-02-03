import tkinter as tk
from tkinter import ttk


def button_func():
    
    # The same thing
    #label.config(text=entry.get())
    label['text'] = entry.get()
    #print(label.configure())
    
    
def button_func2():
    print(string_var.get())
    string_var.set('button_pressed')

#create window
window = tk.Tk()
window.title('Data management')
window.geometry('500x500')

#ttk entry 
entry = ttk.Entry(master = window)
entry.pack()

# test label
label = ttk.Label(master = window,text= 'This is a label')
label.pack()

# button
button  = ttk.Button(master=window, text="Magic button", command= button_func)
button.pack()

##############################################


# TKINTER VARIABLE
# this variable overrites the text of the label
string_var = tk.StringVar()


#ttk entry 
entry2 = ttk.Entry(master = window, textvariable= string_var)
entry2.pack()

# test label
label2 = ttk.Label(master = window,text= 'This is a label',textvariable= string_var)
label2.pack()

# button
button2  = ttk.Button(master=window, text="Magic button2", command= button_func)
button2.pack()


# button
button3  = ttk.Button(master=window, text="Magic button2", command= button_func2)
button3.pack()




# Run the Tkinter main loop
window.mainloop()


# References: 
# https://www.geeksforgeeks.org/python-tkinter-tutorial/

import tkinter as tk
import tkinter.ttk as ttk



root = tk.Tk()
root.title("Hyperspectral App")

# We are making a frame inside the root window
frame = tk.Frame(root)
# Geometry for the frame will be pack()
# Pack puts the widges in rows or columns
frame.pack()

# Create a label for the button
label = tk.Label(frame, text='This is the label text!')
# Place the label in the window using pack geometry
label.pack()


# Create a styling for the button
btnStyle = ttk.Style()
btnStyle.configure(
    'W.TButton',
    font = ('Arial', 14, 'bold'),
    foreground = 'white',
    background = 'red'
)

# Create a button inside the frame
button = ttk.Button(frame, 
                   text='Hyperspectral',
                   style='W.TButton')
# Place the button in the window using pack geometry
button.pack()



# This calls the UI so it keeps displaying in an event loop
root.mainloop()

import tkinter as tk
from tkinter import filedialog
from tkinter import ttk
from tkinter.messagebox import showinfo

filepath = ""

def select_file():
    filename = filedialog.askopenfilename(
        title='Open a file',
        initialdir='/')
    filepath = filename
    """
    showinfo(
        title='Selected File',
        message=filename
    )
    """


root = tk.Tk()
root.title('Open image file to test')
root.resizable(False, False)
root.geometry('300x150')

open_button = ttk.Button(root, text='Open a File', command=select_file)
open_button.pack(expand=True)

root.mainloop()    # run
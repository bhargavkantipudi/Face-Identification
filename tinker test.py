import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
window=tk.Tk()
greeting = tk.Label(text="Hello, Tkinter")
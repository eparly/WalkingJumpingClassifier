import tkinter as tk
import pandas as pd
import matplotlib.pyplot as plt
from tkinter.filedialog import askopenfilename
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from visualizations import pcaScatter
from hdf_to_df import hdf_to_df



class PlotterGUI:
    def __init__(self, master):
        self.master = master
        master.title("Plotter")

        # create a button that opens a file dialog
        self.select_file_button = tk.Button(
            master, text="Select File", command=self.select_file)
        self.select_file_button.pack()

        # create a canvas to display the plot
        self.canvas = tk.Canvas(master, width=400, height=300)
        self.canvas.pack()


    def select_file(self):
        
        # open a file dialog to select a csv file
        filename = askopenfilename(filetypes=[("CSV Files", "*.csv")])

        # read the csv file into a pandas dataframe
        df = pd.read_csv(filename)

        # create a plot of the raw data
        plt.figure(figsize=(6, 4))
        plt.plot(df['Time (s)'], df['Linear Acceleration x (m/s^2)'])
        plt.title('Raw Acceleration Data')
        plt.xlabel('Time (s)')
        plt.ylabel('Linear Acceleration x (m/s^2)')
        plt.grid(True)
        plt.tight_layout()

        # clear the existing plot and update the canvas with the new plot
        self.canvas.delete()
        self.plot_widget = FigureCanvasTkAgg(plt.gcf(), master=self.canvas)
        self.plot_widget.draw()


        # resize the canvas to fit the plot
        self.plot_widget.get_tk_widget().pack(side='top', fill='both', expand=1)
        self.canvas.config(scrollregion=self.canvas.bbox('all'))





if __name__ == '__main__':
    root = tk.Tk()
    app = PlotterGUI(root)
    root.mainloop()

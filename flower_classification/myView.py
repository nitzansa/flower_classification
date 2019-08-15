import flower_classification
import os
import tkinter
from tkinter import Frame, Tk, END, filedialog, Label, ttk, messagebox


class flowersView(Frame):
    def __init__(self, parent):
        Frame.__init__(self, parent)
        self.parent = parent
        parent.configure(background="lightblue")
        self.modelPath = tkinter.StringVar()
        self.folderPath = tkinter.StringVar()
        self.category = tkinter.StringVar()
        self.prediction = {}
        self.selected_category = []
        label1 = Label(parent, text="Select a model to browse:")
        label1.config(bg="lightblue", fg="teal", font=("Cooper Black", 13))
        label1.pack(pady=10)
        tkinter.Entry(textvariable=self.modelPath, width=40).pack()
        tkinter.Button(parent, text="Browse model", command=self.setModelPath, font=("Aharoni", 8)).pack()
        label2 = Label(parent, text="Select a file to open:")
        label2.config(bg="lightblue", fg="teal", font=("Cooper Black", 13))
        label2.pack(pady=10)
        tkinter.Entry(textvariable=self.folderPath, width=40).pack()
        tkinter.Button(parent, text="Browse directory", command=self.setFolderPath, font=("Aharoni", 8)).pack()
        self.predictButton = tkinter.Button(parent, text="Predict", command=self.predict, state='normal')
        self.predictButton.config(font=("Cooper Black", 13), fg="darkslategrey")
        self.predictButton.pack(pady=15)
        classifyNames = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
        self.combo = ttk.Combobox(root, values=classifyNames, width=25)

    def setModelPath(self):
        model_selected = filedialog.askopenfilename(filetype=(("h5 files", ".h5"), ("All files", "*.*")))
        self.modelPath.set(model_selected)

    def setFolderPath(self):
        folder_selected = filedialog.askdirectory()
        self.folderPath.set(folder_selected)

    def predict(self):
        if len(self.modelPath.get()) != 0 and len(self.folderPath.get()) != 0:
            if not os.path.exists(self.modelPath.get()):
                messagebox.showinfo("Alert", "Model does not exist")
                return
            if not os.path.exists(self.folderPath.get()):
                messagebox.showinfo("Alert", "Images folder does not exist")
                return
            self.selectType()
            model = flower_classification.loadWeights(self.modelPath.get())
            self.prediction = flower_classification.predict(model, str(self.folderPath.get() + "/"))
            self.showResults(self.prediction)
        else:
            messagebox.showinfo("Alert", "You must choose any model and images folder first")

    def selectCategory(self):
        categoryName = self.combo.get()
        if os.path.exists('results.csv'):
            self.selected_category = flower_classification.selectByCategory(categoryName, 'results.csv')
        else:
            model = flower_classification.loadWeights(self.modelPath.get())
            self.prediction = flower_classification.predict(model, str(self.folderPath.get() + "/"))
        self.showByCategory(self.selected_category)

    def showResults(self, predict):
        window = tkinter.Toplevel(root)
        window.geometry("400x400")
        frame = tkinter.Frame(window, bd=1, relief='sunken', width=400, height=30)
        scrollbar = tkinter.Scrollbar(frame)
        listbox = tkinter.Listbox(frame, width=400, height=100)
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        listbox.insert(END, "image name, \n classify\n\n")
        for key, value in predict.items():
            listbox.insert(END, "{},\n {}\n\n".format(key, value))
        frame.pack(side='left')
        scrollbar.pack(side='left', fill='y')
        listbox.pack()

    def showByCategory(self, results):
        window = tkinter.Toplevel(root)
        window.geometry("400x400")
        frame = tkinter.Frame(window, bd=1, relief='sunken', width=400, height=30)
        scrollbar = tkinter.Scrollbar(frame)
        listbox = tkinter.Listbox(frame, width=400, height=100)
        listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=listbox.yview)
        listbox.insert(END, "image name\n\n")
        for value in results:
            listbox.insert(END, "{}\n\n".format(value))
        frame.pack(side='right')
        scrollbar.pack(side='right', fill='y')
        listbox.pack()

    def selectType(self):
        select = Label(root, text="Select a flower type")
        select.config(bg="lightblue", fg="teal", font=("Cooper Black", 11))
        select.place(x=415, y=270)
        categoryButton = tkinter.Button(root, text="Select", command=self.selectCategory)
        categoryButton.config(font=("Aharoni", 10))
        categoryButton.place(x=470, y=330)
        self.combo.config(font=("Aharoni", 8))
        self.combo.set("select a flower type")
        self.combo.place(x=420, y=300)
        self.combo.current(0)


root = Tk()
root.title('flowers classify')
root.geometry("600x620+300+30")
flowersView(root)
root.mainloop()
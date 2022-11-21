from utils.timeutils import timeit
import inspect

class Tester:
    logs = None
    def __init__(self, functions, path=None):
        self.functions = []
        for f in functions:
            if type(f) != Binder:
                self.functions.append(Binder(f))
            else:
                self.functions.append(f)

        self.path = path
        self.logs = []
        

    def test(self, *args, loads = None, iters = 10, path = None, 
             save_to_file=False, show_names = None, plot=False, 
             log_results=False, **kwargs):
        
        functions = self.functions
        times = []
        if show_names == None:
            show_names = [str(x) for x in range(len(loads))]
        
        for l, name, it in zip(loads,show_names, range(len(show_names))):
            d = {}
            for f in functions:
                
                  
                d[f"{f.get_name()}_{name}"]= []
     
            times.append(d)
                
        for i in range(iters):
            for f in functions:
                for l, name, it in zip(loads,show_names, range(len(show_names))):
                    times[it][f"{f.get_name()}_{name}"].append(timeit(f, *l[0], **l[1], return_time = True)[0])
        if log_results:
            self.logs.append(times)
            
        if save_to_file:
            if type(save_to_file) == str:
                ...
            else:
                import time
                save_to_file = f"results{str(time.time()).csv}.csv"
            import json
            with open(save_to_file, "w", newline='') as f:
                
                json.dump(times, f)

    
        return times
    
        
    def __call__(self,  *args, save_to_file=False,**kwargs):
        return self.test(*args, save_to_file=save_to_file, **kwargs)
    
    @classmethod
    def plot(self, data, title="chart"):
        
        import json
        import matplotlib.pyplot as plt
        import numpy as np
        if type(data)==str:
            with open(data, "r", newline='') as f:
                data = json.load(f)
                      
        
        grouped = {}
        for d in data:
            for k in d:
                grouped[k.split("_")[0]] = []
            
        for d in data:
            for k,v in d.items():
                grouped[k.split("_")[0]].append(sum(v)/len(v))
            
        for k,v in grouped.items():
            grouped[k] = sorted(v)
        
        
        N = len(list(grouped.values())[0])
        ind = np.arange(N) 
        width = 0.25
        c = 0
        #Bar
        temp = []
        for k,v in grouped.items():
            plt.bar(ind+width*c, v, width, label = k)
            temp.append(v)
            c+=1
        temp = list(map(list, zip(*temp)))
        avgs = []
        for t in temp:
            avgs.append(sum(t)/len(t))
        plt.plot(avgs, color="r")
        plt.title(title+" Bar Comparison with mean trend")
        plt.ylabel("Time(s)")
        plt.xlabel("Load")
        plt.legend()
        plt.show()    
        
        
        for k,v in grouped.items():
            plt.plot(v, label = k)
        plt.title(title+" Comparison and growth")
        plt.ylabel("Time(s)")
        plt.xlabel("Load")
        plt.legend()
        plt.show()    
        
        
        
        for d in data:
            proms = []
            names = []
            for k,v in d.items():
                proms.append((sum(v)/len(v)))
                names.append(k)
            plt.barh(names, proms)
        plt.title(title+" Bars")
        plt.show()
            
    
    
    
class Binder:
    def __init__(self, function, name = None):
        self.function = function
        self.name = name
        
    def get_name(self):
        if self.name is None:
            f = self.function
            return str(inspect.getmodule(f))+"."+f.__qualname__
        return self.name
    
    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)
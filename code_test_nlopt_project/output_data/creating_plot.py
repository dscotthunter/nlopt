import os 
import matplotlib.pyplot as plt
import csv
import numpy as np

fig = plt.figure()
fig.patch.set_facecolor('white')
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 10


f, axarr = plt.subplots(2, 2)


# lambda 1 lasso
data_1 = {}
for file in os.listdir("/Users/davidhunter/nlopt/code_test_nlopt_project/output_data/lasso/"):
	if file.startswith("lambda1_") and "gurobi" not in file:
		print(file[file.find("_")+1:-4])
		data_entry = []
		best_prev = 999990.0
		with open("/Users/davidhunter/nlopt/code_test_nlopt_project/output_data/lasso/"+file) as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if np.log(float(row[0])/ 12241.6863)  < best_prev:
					best_prev = np.log(float(row[0])/ 12241.6863)
					data_entry.append(best_prev)
				else:
					data_entry.append(best_prev)
		data_1[file[file.find("_")+1:-4]] = data_entry

		#axarr[0,0].plot(data_entry[:400], linewidth=1.0, markersize=0.0)
		
		axarr[0,0].set_ylim([0.0,1.0])
		axarr[0,0].set_title("Lasso $\lambda = 1$")


data_100 = {}
for file in os.listdir("/Users/davidhunter/nlopt/code_test_nlopt_project/output_data/lasso/"):
	if file.startswith("lambda100_") and "gurobi" not in file:
		print(file[file.find("_")+1:-4])
		data_entry = []
		best_prev = 999990.0
		with open("/Users/davidhunter/nlopt/code_test_nlopt_project/output_data/lasso/"+file) as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if np.log(float(row[0])/ 13244.9599282)  < best_prev:
					best_prev = np.log(float(row[0])/ 13244.9599282)
					data_entry.append(best_prev)
				else:
					data_entry.append(best_prev)
		data_100[file[file.find("_")+1:-4]] = data_entry

		#axarr[0,1].plot(data_entry[:400], linewidth=1.0, markersize=0.0)
		
		axarr[0,1].set_ylim([0.0,1.0])
		axarr[0,1].set_title("Lasso $\lambda = 100$")

data_10000 = {}
for file in os.listdir("/Users/davidhunter/nlopt/code_test_nlopt_project/output_data/lasso/"):
	if file.startswith("lambda10000_") and "gurobi" not in file:
		print(file[file.find("_")+1:-4])
		data_entry = []
		best_prev = 999990.0
		with open("/Users/davidhunter/nlopt/code_test_nlopt_project/output_data/lasso/"+file) as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if np.log(float(row[0])/ 36679.7521)  < best_prev:
					best_prev = np.log(float(row[0])/ 36679.7521)
					data_entry.append(best_prev)
				else:
					data_entry.append(best_prev)
		data_10000[file[file.find("_")+1:-4]] = data_entry

		#axarr[1,0].plot(data_entry[:50], linewidth=1.0, markersize=0.0)
		
		axarr[1,0].set_ylim([0.0,1.0])
		axarr[1,0].set_title("Lasso $\lambda = 10000$")


data_logistic = {}
for file in os.listdir("/Users/davidhunter/nlopt/code_test_nlopt_project/output_data/logistic/"):
	if file.startswith("lambda1_") and "gurobi" not in file:
		print(file[file.find("_")+1:-4])
		data_entry = []
		best_prev = 999990.0
		with open("/Users/davidhunter/nlopt/code_test_nlopt_project/output_data/logistic/"+file) as csvfile:
			reader = csv.reader(csvfile)
			for row in reader:
				if np.log(float(row[0])/ 59.843537754)  < best_prev:
					best_prev = np.log(float(row[0])/ 59.843537754)
					data_entry.append(best_prev)
				else:
					data_entry.append(best_prev)
		data_logistic[file[file.find("_")+1:-4]] = data_entry

		#axarr[1,1].plot(data_entry[:2000], linewidth=1.0, markersize=0.0)
		
		axarr[1,1].set_ylim([0.0,1.0])
		axarr[1,1].set_title("Logistic $\lambda = 1$")		
# axarr[0, 0].plot(x, y)
# axarr[0, 0].set_title('Axis [0,0]')
# axarr[0, 1].scatter(x, y)
# axarr[0, 1].set_title('Axis [0,1]')
# axarr[1, 0].plot(x, y ** 2)
# axarr[1, 0].set_title('Axis [1,0]')
# axarr[1, 1].scatter(x, y ** 2)
# axarr[1, 1].set_title('Axis [1,1]')
colors=['b','g','r', 'c']
entry_replace = {"owlqn5": "OWL-QN $M=5$", "owlqn2": "OWL-QN $M=2$", "owlqn10": "OWL-QN $M=10$", "ccsa": "CCSA-Q"}
for index, entry in enumerate(["owlqn2", "owlqn5", "owlqn10", "ccsa"]):
	axarr[0,0].plot(data_1[entry][:400], linewidth=1.0, markersize=0.0, color=colors[index])
	axarr[0,1].plot(data_100[entry][:400], linewidth=1.0, markersize=0.0,color= colors[index], label=entry_replace[entry])
	axarr[1,0].plot(data_10000[entry][:150], linewidth=1.0, markersize=0.0, color=colors[index])
	axarr[1,1].plot(data_logistic[entry][:2000], linewidth=1.0, markersize=0.0, color=colors[index])

axarr[0,1].legend()
plt.show()

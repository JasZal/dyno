import os.path
import re
import ast

import matplotlib.pyplot as plt

# Function to read .txt files
def read_data(filename):
    data = {}
    with open(filename, "r") as f:
        for line in f:
            if "=" in line:
                name, values = line.split("=")
                name = name.strip()
                values = values.strip()
                if "[" in values and " " in values and "," not in values:
                    values = re.sub(r"\s+", ", ", values)

                values = ast.literal_eval(values)
                data[name] = values
    return data


# Set fixed colors
colors = {
    "LBW50": "blue",
    "PCS50": "orange",
    "UIS50": "green",
}

# Utilities of LBW, PCS and UIS using Dyno
utility_dyno = read_data("./results/log_reg_utility.txt")
eps1 = utility_dyno["eps"]

for key, values in utility_dyno.items():
    if key != "eps":
        plt.plot(eps1, values, color=colors.get(key, None), linestyle="-", label=key)


# Utilities of LDP approach
utility_ldp = read_data("./results/log_reg_utility_ldp.txt")
eps2 = utility_ldp["eps"]


for key, values in utility_ldp.items():
    if key != "eps":
        plt.plot(eps2, values, color=colors.get(key, None), linestyle="--", label=None)


plt.xlabel("eps_max")
plt.ylabel("accuracy")
plt.title("Model utility")
plt.legend()
plt.xlim(0, 8)

plt.savefig("./results/log_reg_utility.pdf", bbox_inches="tight")  
plt.show()

# Only plot NHANES utilities if file has been created
if os.path.exists("./results/log_reg_utility_nhanes.txt"):

    # Utility of Nhanes using Dyno
    utility_nhanes = read_data("./results/log_reg_utility_nhanes.txt")
    eps = utility_nhanes["eps"]
    for key, values in utility_nhanes.items():
        if key != "eps":
            plt.plot(eps, values, linestyle="-", label=f"it = {key}")

    # Utilities of LDP approach
    utility_nhanes_ldp = read_data("./results/log_reg_utility_nhanes_ldp.txt")
    eps_nhanes_ldp = utility_nhanes_ldp["eps"]
    for key, values in utility_nhanes_ldp.items():
        if key != "eps":
            plt.plot(eps_nhanes_ldp, values, color="black", linestyle="--", label="LDP")

    plt.xlabel("eps_max")
    plt.ylabel("accuracy")
    plt.title("Model utility Nhanes")
    plt.legend()
    plt.xlim(0, 8)

    plt.savefig("./results/log_reg_utility_nhanes.pdf", bbox_inches="tight")  
    plt.show()


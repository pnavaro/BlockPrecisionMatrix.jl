# -*- coding: utf-8 -*-
include("src/utils.jl")

# + {"endofcell": "--"}
using Plots, CSV

# Load temperature data
data = CSV.read(download("https://raw.githubusercontent.com/eriklindernoren/ML-From-Scratch/master/mlfromscratch/data/TempLinkoping2016.txt"), delim="\t");
# -

time = reshape(collect(data.time), size(data)[1], 1)
temp = collect(data.temp);

X = time # fraction of the year [0, 1]
y = temp;

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

scatter( X_train, y_train)
scatter!( X_test, y_test)
# --

# +
include("src/lasso.jl")
model = LassoRegression(X_train, y_train,
                        degree=15, 
                        reg_factor=0.05,
                        learning_rate=0.001,
                        n_iterations=4000)




# +
y_pred = predict(model, X_test)

scatter( X_test, y_pred)
scatter!(X_test, y_test)

# +
include("src/lasso.jl")

fitreg = LassoRegression(degree=15, 
                         reg_factor=0.05,
                         learning_rate=0.001,
                         n_iterations=4000)

y_line = fit_and_predict(fitreg, X, y)

plot(X, y_line)
scatter!(X, y, markersize=1)
# -



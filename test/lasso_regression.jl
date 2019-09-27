# +
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

size(y)

# +
include("../src/utils.jl")
include("../src/lasso.jl")

model = LassoRegression(degree=10, 
                        reg_factor=0.05,
                        learning_rate=0.001,
                        n_iterations=4000)

y_pred = fit_and_predict(model, X, y);

plot(X, y_pred)
scatter!(X, y)
# -



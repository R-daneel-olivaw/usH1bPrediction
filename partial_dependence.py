from sklearn.ensemble.partial_dependence import partial_dependence, plot_partial_dependence
import model_generator as mg

# naics_2007_us_code
# us_economic_sector
X_columns = ['employer_name_processed', 'employer_state_processed', 'country_of_citizenship_processed',
                          'pw_soc_code_processed', 'pw_amount_9089_processed']
my_model, X = mg.get_trained_model()
# pr = my_model.predict([35371, 51, 76, 172051, 47923])
pr = my_model.predict(X.head(20))
print(X.head(20))
print("Prediction")
print(pr)

# plot_partial_dependence(h1b_model, features=[0, 1, 2, 3, 4],  # column numbers of plots we want to show
#                         X=X,            # raw predictors data.
#                         feature_names=['Employer', 'Employer_State', 'Country_Citizenship', 'Job Type Code',
#                                        'Salary'],  # labels on graphs
#                         grid_resolution=10)  # number of values to plot on x axis

import sklearn
from prepare_datasets import get_pruned_df
from pprint import pprint
from sklearn.model_selection import train_test_split

prepared_df = get_pruned_df()
# pprint(prepared_df)

X_train, X_test, Y_train, Y_test = train_test_split(prepared_df.drop(columns=['real_temp']), prepared_df['real_temp'], shuffle=False, random_state=0, test_size=1000)
pprint(X_train)

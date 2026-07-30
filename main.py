from src.data_ingestion import data_ingestion
from src.data_preprocessing import preprocessing
from src.model_building import model_build
def main():
    #Step 1: Data Ingestion 
    df = data_ingestion()
    print(df.shape)

    #Step 2: Data Preprocessing
    X_train,X_test,y_train,y_test = preprocessing(df)
    print(X_train.shape)

    #Step 3: Model Building
    model,accuracy = model_build(X_train,X_test,y_train,y_test)

main()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df= pd.read_csv('housing_2.csv', header=None, delimiter=r"\s+", names=column_names)



#Nhan xet ve cac du lieu
# for label in column_names[:-1]:
#     plt.scatter(df[label],df['MEDV'])
#     plt.xlabel(label)
#     plt.ylabel('Price')
#     plt.show()



print('Kich thuoc tap tin=', df.shape)
print(' \nKiem tra cac tap du lieu co chua du lieu nan:')
for lable in column_names[:-1]:
    print(f'{lable}: {(df[lable].isnull().sum())}')



def check_datatypes(dataset):
    return dataset.dtypes

print("Kiem tra loai du lieu trong tap data:")
print(check_datatypes(df))


from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


x = df[df.columns[:-1]]
y  = df[df.columns[-1]]


def rfc_feature_selection(dataset,target):  #thực hiện việc lựa chọn đặc trưng quan trọng sử dụng mô hình RandomForestRegressor.
    X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.3, random_state=42)
    rfc = RandomForestRegressor(random_state=42)
    rfc.fit(X_train, y_train)
    y_pred = rfc.predict(X_test)
    rfc_importances = pd.Series(rfc.feature_importances_, index=dataset.columns).sort_values().tail(10) #tính toán độ quan trọng của từng đặc trưng (feature importance) sắp xếp tăng dần xong lấy 10 giá trị cao nhất
    print(rfc_importances)
    rfc_importances.plot(kind='bar')  #trực quan hóa độ quan trọng của các đặc trưng bằng biểu đồ cột.                
    plt.show()

print(rfc_feature_selection(x,y))

# Chon ra thuoc dac trung co quan trong nhất ảnh hương nhiều nhất đến kết quả đầu ra của mô hình 
x= x[['CRIM','DIS','RM','LSTAT']]

mms= MinMaxScaler()
x = pd.DataFrame(mms.fit_transform(x), columns=x.columns)

#Lay hai tap train test ngẫu nhiên theo tỷ lệ train 7 test 3
xtrain,xtest,ytrain,ytest= train_test_split(x,y,test_size=0.3,random_state=4)


#Linear Regression
lr=LinearRegression()

lr.fit(xtrain, ytrain)

coefficients=pd.DataFrame([xtrain.columns, lr.coef_]).T
coefficients=coefficients.rename(columns={0:'Attributes',1:'Coefficients'})
print(coefficients) #in ra hệ số


y_pred=lr.predict(xtrain)

#training data
print(f'Cac thong so danh gia chat luong mo hinh Linear Regrssion(train)')
print("R^2: ",metrics.r2_score(ytrain, y_pred))
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytrain, y_pred))*(len(ytrain)-1)/(len(ytrain)-xtrain.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytrain, y_pred))
print("MSE: ", metrics.mean_squared_error(ytrain, y_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytrain, y_pred)))

plt.scatter(ytrain, y_pred)
plt.xlabel("Actual Price")  # giá thực tế 
plt.ylabel("Predicted Price")  # giá dự đoán
plt.title("Predicted Vs Actual Prices", fontsize=15)
plt.show()



#test data
ytest_pred=lr.predict(xtest)

from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error,max_error

lin_acc=metrics.r2_score(ytest, ytest_pred)

print(f'Cac thong so danh gia chat luong mo hinh Linear Regrssion(test)')
print("R^2: ",lin_acc)
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytest, ytest_pred))*(len(ytest)-1)/(len(ytest)-xtest.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytest, ytest_pred))
print("MSE: ", metrics.mean_squared_error(ytest, ytest_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytest, ytest_pred)))

plt.scatter(ytest, ytest_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices", fontsize=15)
plt.show()


#SVM
from sklearn.svm import SVR
svm_reg = SVR(C=150)  #tham số C điều chuẩn, quyết định độ nghiêng của ranh giới quyết định
svm_reg.fit(xtrain,ytrain)

#train
y_pred=svm_reg.predict(xtrain)

print(f'Cac thong so danh gia chat luong mo hinh SVM cho bai toan (train)')
print("R^2: ",metrics.r2_score(ytrain, y_pred))
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytrain, y_pred))*(len(ytrain)-1)/(len(ytrain)-xtrain.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytrain, y_pred))
print("MSE: ", metrics.mean_squared_error(ytrain, y_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytrain, y_pred)))

plt.scatter(ytrain, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices SVMtrain", fontsize=15)
plt.show()

#test
y_pred=svm_reg.predict(xtest)


svm_acc =r2_score(ytest,y_pred)

print(f'Cac thong so danh gia chat luong mo hinh SVM cho bai toan (test)')
print("R^2: ",svm_acc)
print("Adusted R^2: ", 1-(1-metrics.r2_score(ytest, ytest_pred))*(len(ytest)-1)/(len(ytest)-xtest.shape[1]-1))
print("MAE: ", metrics.mean_absolute_error(ytest, ytest_pred))
print("MSE: ", metrics.mean_squared_error(ytest, ytest_pred))
print("RMSE: ",np.sqrt(metrics.mean_squared_error(ytest, ytest_pred)))


plt.scatter(ytest, y_pred)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted Vs Actual Prices SVMtest", fontsize=15)
plt.show()

models=pd.DataFrame({
    'Model':['Linear Regression', 'Support Vector Machine'],
    'R_squared Score':[lin_acc*100,svm_acc*100],
})
print(models.sort_values(by='R_squared Score', ascending=False))

# Chon mo hinh tot nhat 
print('---===---'*15)
print('Best model cho bai toan du doan ra nha voi tap du lieu nay la SVM voi do chinh xac 81%')
print('---===---'*15)
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 데이터 불러오기
bc = load_breast_cancer()
print(type(dir(bc)))  

#bc에 어떤 정보들이 담겨있는지 확인하기
print("bc에 담겨진 정보 : ", bc.keys())

#Feature Data 지정하기
bc_data = bc.data

#bc_data 크기 확인해보기(배열의 형상정보 출력하기)
print( bc_data.shape )

#샘플로 bc_data에서 하나의 데이터만 확인해 보기)
print( bc_data[0] )

bc_label = bc.target
print(bc_label.shape)
print(bc_label)

#라벨의 이름을 출력해 봅시다. 
print(bc.target_names)
# benign : (of a disease) not harmful in effect.
# malignant : (of a disease) very virulent or infectious.

#데이터의 설명이 담겨있는 변수를 출력해 봅시다. 
print(bc.DESCR)

#feature에 대한 설명이 담긴 변수를 출력해 봅시다. 
print(bc.feature_names)

# 데이터셋 파일이 저장된 경로를 출력해 봅시다. 
print(bc.filename)
import pandas as pd

#유방암 데이터셋을 pandas가 제공하는 DataFrame이라는 자료형으로 변환하기 
bc_df = pd.DataFrame(data=bc_data, columns=bc.feature_names)

#정답 데이터를 `label` 이라는 컬럼으로 추가 해 주기 
bc_df["label"] = bc.target

# sklearn.model_selection 패키지의 train_test_split을 활용
from sklearn.model_selection import train_test_split

# trainig dataset과 test dataset을 간단히 분리해 봅시다.
X_train, X_test, y_train, y_test = train_test_split(bc_data, 
                                                    bc_label, # bc.target과 같음. 정답 데이터 
                                                    test_size=0.2, 
                                                    random_state=7)

print('X_train 개수: ', len(X_train), ', X_test 개수: ', len(X_test))
print('y_train 개수: ', len(y_train), ', y_test 개수: ', len(y_test))

# 의사결정트리 사용해보기 
#사이킷런의 의사결정트리를 import 합니다. 
from sklearn.tree import DecisionTreeClassifier

#의사결정 트리를 선언합니다. 
DT = DecisionTreeClassifier(random_state=64)
print(DT._estimator_type)

#훈련데이터로 의사결정트리를 학습합니다. 
DT.fit(X_train, y_train)

# 랜덤포레스트 
#사이킷런의 랜덤포레스트를 import 합니다. 
from sklearn.ensemble import RandomForestClassifier

#랜덤 포레스트를 생성합니다. 
RF = RandomForestClassifier(random_state=64)
print(RF._estimator_type)

#훈련데이터로 랜덤포레스트를 학습합니다. 
RF.fit(X_train, y_train)

# SVM
#사이킷런의 svm을 import 합니다. 
from sklearn import svm

#svm을 생성합니다.
SVM = svm.SVC()
print(SVM._estimator_type)

#훈련데이터로 svm분류모델을 학습합니다. 
SVM.fit(X_train, y_train)

# SGD 
#사이킷런의 SGD Classifier 모델을 import 합니다. 
from sklearn.linear_model import SGDClassifier

#sgd classifier를 생성합니다. 
SGD = SGDClassifier()
print(DT._estimator_type)

#훈련데이터로 SGC분류기를 학습합니다. 
SGD.fit(X_train, y_train)

#사이킷런의 로지스틱 회귀를 import 합니다. 
from sklearn.linear_model import LogisticRegression

#의사결정 트리를 선언합니다. 
LOGI = LogisticRegression()
print(LOGI._estimator_type)

#훈련데이터로 로지스틱 회귀를 학습합니다. 
LOGI.fit(X_train, y_train)

# sklearn.metrics의 accuaracy_score함수를 import 해 주세요
from sklearn.metrics import accuracy_score
model_list = [DT, RF, SVM, SGD, LOGI]

y_pred_list = []

for model in model_list:
  y_pred_list.append(model.predict(X_test))
  
acc_list = []

for y_pred in y_pred_list : 
  acc_list.append(accuracy_score(y_test, y_pred))
  
print(acc_list)

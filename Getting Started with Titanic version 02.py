# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:54:28.547730Z","iopub.execute_input":"2021-07-07T11:54:28.548112Z","iopub.status.idle":"2021-07-07T11:54:28.554160Z","shell.execute_reply.started":"2021-07-07T11:54:28.548080Z","shell.execute_reply":"2021-07-07T11:54:28.553000Z"}}
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt # data visualization
import seaborn as sns # data visualization

import warnings
warnings.filterwarnings('ignore')

plt.style.use('ggplot')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:54:36.188696Z","iopub.execute_input":"2021-07-07T11:54:36.189206Z","iopub.status.idle":"2021-07-07T11:54:36.214185Z","shell.execute_reply.started":"2021-07-07T11:54:36.189172Z","shell.execute_reply":"2021-07-07T11:54:36.212832Z"}}
train_df= pd.read_csv('../input/titanic/train.csv')
test_df = pd.read_csv('../input/titanic/test.csv')

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:54:48.664950Z","iopub.execute_input":"2021-07-07T11:54:48.665481Z","iopub.status.idle":"2021-07-07T11:54:48.670298Z","shell.execute_reply.started":"2021-07-07T11:54:48.665439Z","shell.execute_reply":"2021-07-07T11:54:48.669281Z"}}
train = train_df.copy()
test= test_df.copy()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:55:08.845837Z","iopub.execute_input":"2021-07-07T11:55:08.846222Z","iopub.status.idle":"2021-07-07T11:55:08.867302Z","shell.execute_reply.started":"2021-07-07T11:55:08.846183Z","shell.execute_reply":"2021-07-07T11:55:08.866321Z"}}
print(train.info())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:55:40.961537Z","iopub.execute_input":"2021-07-07T11:55:40.962023Z","iopub.status.idle":"2021-07-07T11:55:40.984694Z","shell.execute_reply.started":"2021-07-07T11:55:40.961981Z","shell.execute_reply":"2021-07-07T11:55:40.983113Z"}}
print(test.info())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:56:44.237346Z","iopub.execute_input":"2021-07-07T11:56:44.237765Z","iopub.status.idle":"2021-07-07T11:56:44.276112Z","shell.execute_reply.started":"2021-07-07T11:56:44.237731Z","shell.execute_reply":"2021-07-07T11:56:44.274630Z"}}
print(train.describe())


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:56:53.067857Z","iopub.execute_input":"2021-07-07T11:56:53.068237Z","iopub.status.idle":"2021-07-07T11:56:53.095561Z","shell.execute_reply.started":"2021-07-07T11:56:53.068208Z","shell.execute_reply":"2021-07-07T11:56:53.094757Z"}}
print(test.describe())


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:57:01.427148Z","iopub.execute_input":"2021-07-07T11:57:01.427704Z","iopub.status.idle":"2021-07-07T11:57:01.436829Z","shell.execute_reply.started":"2021-07-07T11:57:01.427657Z","shell.execute_reply":"2021-07-07T11:57:01.435195Z"}}
print(train.isnull().sum())


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:58:08.908251Z","iopub.execute_input":"2021-07-07T11:58:08.908970Z","iopub.status.idle":"2021-07-07T11:58:08.919777Z","shell.execute_reply.started":"2021-07-07T11:58:08.908910Z","shell.execute_reply":"2021-07-07T11:58:08.918715Z"}}
print(test.isnull().sum())

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:58:23.201010Z","iopub.execute_input":"2021-07-07T11:58:23.201416Z","iopub.status.idle":"2021-07-07T11:58:23.207281Z","shell.execute_reply.started":"2021-07-07T11:58:23.201382Z","shell.execute_reply":"2021-07-07T11:58:23.206326Z"}}
print(train.columns)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:58:32.139689Z","iopub.execute_input":"2021-07-07T11:58:32.140053Z","iopub.status.idle":"2021-07-07T11:58:32.146136Z","shell.execute_reply.started":"2021-07-07T11:58:32.140023Z","shell.execute_reply":"2021-07-07T11:58:32.145138Z"}}
print(test.columns)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:58:47.947316Z","iopub.execute_input":"2021-07-07T11:58:47.947715Z","iopub.status.idle":"2021-07-07T11:58:47.956830Z","shell.execute_reply.started":"2021-07-07T11:58:47.947682Z","shell.execute_reply":"2021-07-07T11:58:47.955364Z"}}
train.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
test.drop(columns= ['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace= True)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:58:54.015707Z","iopub.execute_input":"2021-07-07T11:58:54.016100Z","iopub.status.idle":"2021-07-07T11:58:54.023838Z","shell.execute_reply.started":"2021-07-07T11:58:54.016060Z","shell.execute_reply":"2021-07-07T11:58:54.022719Z"}}
train['Age'].median()

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2021-07-07T11:59:51.848561Z","iopub.execute_input":"2021-07-07T11:59:51.848982Z","iopub.status.idle":"2021-07-07T11:59:51.857461Z","shell.execute_reply.started":"2021-07-07T11:59:51.848951Z","shell.execute_reply":"2021-07-07T11:59:51.856433Z"}}
train['Embarked'].mode()[0]

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:00:13.492313Z","iopub.execute_input":"2021-07-07T12:00:13.492842Z","iopub.status.idle":"2021-07-07T12:00:13.501081Z","shell.execute_reply.started":"2021-07-07T12:00:13.492808Z","shell.execute_reply":"2021-07-07T12:00:13.500158Z"}}
train['Age'].fillna(train['Age'].median(), inplace=True)
train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:00:21.383920Z","iopub.execute_input":"2021-07-07T12:00:21.384466Z","iopub.status.idle":"2021-07-07T12:00:21.391934Z","shell.execute_reply.started":"2021-07-07T12:00:21.384422Z","shell.execute_reply":"2021-07-07T12:00:21.390725Z"}}
print(train.isnull().sum())

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:00:43.631542Z","iopub.execute_input":"2021-07-07T12:00:43.632460Z","iopub.status.idle":"2021-07-07T12:00:43.643184Z","shell.execute_reply.started":"2021-07-07T12:00:43.632398Z","shell.execute_reply":"2021-07-07T12:00:43.640343Z"}}
test['Age'].median()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:00:49.799654Z","iopub.execute_input":"2021-07-07T12:00:49.800064Z","iopub.status.idle":"2021-07-07T12:00:49.807869Z","shell.execute_reply.started":"2021-07-07T12:00:49.800032Z","shell.execute_reply":"2021-07-07T12:00:49.806637Z"}}
test['Fare'].median()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:01:23.273584Z","iopub.execute_input":"2021-07-07T12:01:23.273966Z","iopub.status.idle":"2021-07-07T12:01:23.281241Z","shell.execute_reply.started":"2021-07-07T12:01:23.273933Z","shell.execute_reply":"2021-07-07T12:01:23.279766Z"}}
test['Age'].fillna(test['Age'].median(), inplace=True)
test['Fare'].fillna(test['Fare'].median(), inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:01:29.832326Z","iopub.execute_input":"2021-07-07T12:01:29.832803Z","iopub.status.idle":"2021-07-07T12:01:29.843032Z","shell.execute_reply.started":"2021-07-07T12:01:29.832762Z","shell.execute_reply":"2021-07-07T12:01:29.841411Z"}}
print(test.isnull().sum())

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:01:40.805373Z","iopub.execute_input":"2021-07-07T12:01:40.805924Z","iopub.status.idle":"2021-07-07T12:01:40.814098Z","shell.execute_reply.started":"2021-07-07T12:01:40.805890Z","shell.execute_reply":"2021-07-07T12:01:40.813153Z"}}
train['Survived'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:02:20.131002Z","iopub.execute_input":"2021-07-07T12:02:20.131504Z","iopub.status.idle":"2021-07-07T12:02:20.143972Z","shell.execute_reply.started":"2021-07-07T12:02:20.131463Z","shell.execute_reply":"2021-07-07T12:02:20.141802Z"}}
train['Pclass'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:02:30.903083Z","iopub.execute_input":"2021-07-07T12:02:30.903498Z","iopub.status.idle":"2021-07-07T12:02:30.914296Z","shell.execute_reply.started":"2021-07-07T12:02:30.903463Z","shell.execute_reply":"2021-07-07T12:02:30.912427Z"}}
train['Sex'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:02:38.685723Z","iopub.execute_input":"2021-07-07T12:02:38.686190Z","iopub.status.idle":"2021-07-07T12:02:38.695833Z","shell.execute_reply.started":"2021-07-07T12:02:38.686111Z","shell.execute_reply":"2021-07-07T12:02:38.694649Z"}}
train['SibSp'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:02:49.666661Z","iopub.execute_input":"2021-07-07T12:02:49.667000Z","iopub.status.idle":"2021-07-07T12:02:49.675406Z","shell.execute_reply.started":"2021-07-07T12:02:49.666971Z","shell.execute_reply":"2021-07-07T12:02:49.674347Z"}}
train['Parch'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:07:32.440827Z","iopub.execute_input":"2021-07-07T12:07:32.441227Z","iopub.status.idle":"2021-07-07T12:07:32.450580Z","shell.execute_reply.started":"2021-07-07T12:07:32.441194Z","shell.execute_reply":"2021-07-07T12:07:32.449526Z"}}
train['Embarked'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:07:53.407535Z","iopub.execute_input":"2021-07-07T12:07:53.407987Z","iopub.status.idle":"2021-07-07T12:07:53.417354Z","shell.execute_reply.started":"2021-07-07T12:07:53.407952Z","shell.execute_reply":"2021-07-07T12:07:53.416369Z"}}
test['Pclass'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:08:05.885148Z","iopub.execute_input":"2021-07-07T12:08:05.885541Z","iopub.status.idle":"2021-07-07T12:08:05.897902Z","shell.execute_reply.started":"2021-07-07T12:08:05.885510Z","shell.execute_reply":"2021-07-07T12:08:05.896928Z"}}
test['Sex'].value_counts()test['SibSp'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:08:12.073390Z","iopub.execute_input":"2021-07-07T12:08:12.073841Z","iopub.status.idle":"2021-07-07T12:08:12.083720Z","shell.execute_reply.started":"2021-07-07T12:08:12.073807Z","shell.execute_reply":"2021-07-07T12:08:12.082724Z"}}
test['SibSp'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:08:18.825214Z","iopub.execute_input":"2021-07-07T12:08:18.825601Z","iopub.status.idle":"2021-07-07T12:08:18.835803Z","shell.execute_reply.started":"2021-07-07T12:08:18.825555Z","shell.execute_reply":"2021-07-07T12:08:18.834456Z"}}
test['Parch'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:08:26.411219Z","iopub.execute_input":"2021-07-07T12:08:26.411705Z","iopub.status.idle":"2021-07-07T12:08:26.422959Z","shell.execute_reply.started":"2021-07-07T12:08:26.411667Z","shell.execute_reply":"2021-07-07T12:08:26.421855Z"}}
test['Embarked'].value_counts()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:08:37.944799Z","iopub.execute_input":"2021-07-07T12:08:37.945166Z","iopub.status.idle":"2021-07-07T12:08:38.126942Z","shell.execute_reply.started":"2021-07-07T12:08:37.945135Z","shell.execute_reply":"2021-07-07T12:08:38.125612Z"}}
plt.figure(figsize=(8,6))
sns.countplot(x='Survived', data= train)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:08:58.779025Z","iopub.execute_input":"2021-07-07T12:08:58.779578Z","iopub.status.idle":"2021-07-07T12:08:58.909894Z","shell.execute_reply.started":"2021-07-07T12:08:58.779528Z","shell.execute_reply":"2021-07-07T12:08:58.908889Z"}}
plt.figure(figsize=(8,6))
sns.countplot(x='Sex', data= train)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:09:06.394971Z","iopub.execute_input":"2021-07-07T12:09:06.395418Z","iopub.status.idle":"2021-07-07T12:09:06.550535Z","shell.execute_reply.started":"2021-07-07T12:09:06.395378Z","shell.execute_reply":"2021-07-07T12:09:06.549212Z"}}
plt.figure(figsize=(8,6))
sns.countplot(x='Survived', hue='Sex', data= train)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:09:31.566796Z","iopub.execute_input":"2021-07-07T12:09:31.567405Z","iopub.status.idle":"2021-07-07T12:09:31.783770Z","shell.execute_reply.started":"2021-07-07T12:09:31.567361Z","shell.execute_reply":"2021-07-07T12:09:31.782477Z"}}
plt.figure(figsize=(8,6))
sns.countplot(x='Survived', hue='Pclass', data= train)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:10:19.276156Z","iopub.execute_input":"2021-07-07T12:10:19.276635Z","iopub.status.idle":"2021-07-07T12:10:19.521397Z","shell.execute_reply.started":"2021-07-07T12:10:19.276573Z","shell.execute_reply":"2021-07-07T12:10:19.519750Z"}}
plt.figure(figsize=(8,6))
sns.boxplot(x='Survived', y= 'Age', hue='Sex', data= train)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:10:49.445887Z","iopub.execute_input":"2021-07-07T12:10:49.446279Z","iopub.status.idle":"2021-07-07T12:10:49.619497Z","shell.execute_reply.started":"2021-07-07T12:10:49.446245Z","shell.execute_reply":"2021-07-07T12:10:49.617937Z"}}
plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass', y= 'Fare', data= train)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:11:29.064532Z","iopub.execute_input":"2021-07-07T12:11:29.064934Z","iopub.status.idle":"2021-07-07T12:11:29.199901Z","shell.execute_reply.started":"2021-07-07T12:11:29.064894Z","shell.execute_reply":"2021-07-07T12:11:29.198643Z"}}
plt.figure(figsize=(8,6))
sns.countplot(x='Sex', data= test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:12:01.034938Z","iopub.execute_input":"2021-07-07T12:12:01.035357Z","iopub.status.idle":"2021-07-07T12:12:01.222567Z","shell.execute_reply.started":"2021-07-07T12:12:01.035316Z","shell.execute_reply":"2021-07-07T12:12:01.221061Z"}}
plt.figure(figsize=(8,6))
sns.boxplot(x='Pclass', y= 'Fare', data= test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:12:11.283570Z","iopub.execute_input":"2021-07-07T12:12:11.284468Z","iopub.status.idle":"2021-07-07T12:12:11.856067Z","shell.execute_reply.started":"2021-07-07T12:12:11.284402Z","shell.execute_reply":"2021-07-07T12:12:11.854991Z"}}
train.plot(kind='box', figsize= (10,8))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:14:27.409634Z","iopub.execute_input":"2021-07-07T12:14:27.410139Z","iopub.status.idle":"2021-07-07T12:14:27.439313Z","shell.execute_reply.started":"2021-07-07T12:14:27.410100Z","shell.execute_reply":"2021-07-07T12:14:27.437781Z"}}
cols= ['Age', 'SibSp', 'Parch', 'Fare']

train[cols]= train[cols].clip(lower= train[cols].quantile(0.15), upper= train[cols].quantile(0.85), axis=1)

train.drop(columns=['Parch'], axis=1, inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:14:34.899845Z","iopub.execute_input":"2021-07-07T12:14:34.900387Z","iopub.status.idle":"2021-07-07T12:14:35.119138Z","shell.execute_reply.started":"2021-07-07T12:14:34.900347Z","shell.execute_reply":"2021-07-07T12:14:35.118165Z"}}
train.plot(kind='box', figsize= (10,8)) 

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:15:13.216389Z","iopub.execute_input":"2021-07-07T12:15:13.216795Z","iopub.status.idle":"2021-07-07T12:15:13.435509Z","shell.execute_reply.started":"2021-07-07T12:15:13.216759Z","shell.execute_reply":"2021-07-07T12:15:13.434090Z"}}
test.plot(kind='box', figsize= (10,8))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:15:32.339344Z","iopub.execute_input":"2021-07-07T12:15:32.339730Z","iopub.status.idle":"2021-07-07T12:15:32.364798Z","shell.execute_reply.started":"2021-07-07T12:15:32.339696Z","shell.execute_reply":"2021-07-07T12:15:32.363280Z"}}
test[cols]= test[cols].clip(lower= test[cols].quantile(0.15), upper= test[cols].quantile(0.85), axis=1)

test.drop(columns=['Parch'], axis=1, inplace=True)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:15:42.162165Z","iopub.execute_input":"2021-07-07T12:15:42.162615Z","iopub.status.idle":"2021-07-07T12:15:42.361739Z","shell.execute_reply.started":"2021-07-07T12:15:42.162565Z","shell.execute_reply":"2021-07-07T12:15:42.360248Z"}}
test.plot(kind='box', figsize= (10,8))  

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:16:22.335683Z","iopub.execute_input":"2021-07-07T12:16:22.336105Z","iopub.status.idle":"2021-07-07T12:16:22.354233Z","shell.execute_reply.started":"2021-07-07T12:16:22.336072Z","shell.execute_reply":"2021-07-07T12:16:22.353406Z"}}
train= pd.get_dummies(train, columns=['Pclass', 'Sex', 'Embarked' ], drop_first= True)

test= pd.get_dummies(test, columns=['Pclass', 'Sex', 'Embarked' ], drop_first= True)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:16:32.255510Z","iopub.execute_input":"2021-07-07T12:16:32.256186Z","iopub.status.idle":"2021-07-07T12:16:32.271324Z","shell.execute_reply.started":"2021-07-07T12:16:32.256145Z","shell.execute_reply":"2021-07-07T12:16:32.270501Z"}}
train.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:16:41.093645Z","iopub.execute_input":"2021-07-07T12:16:41.094361Z","iopub.status.idle":"2021-07-07T12:16:41.108699Z","shell.execute_reply.started":"2021-07-07T12:16:41.094317Z","shell.execute_reply":"2021-07-07T12:16:41.107718Z"}}
test.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:17:47.238529Z","iopub.execute_input":"2021-07-07T12:17:47.239007Z","iopub.status.idle":"2021-07-07T12:17:47.246753Z","shell.execute_reply.started":"2021-07-07T12:17:47.238970Z","shell.execute_reply":"2021-07-07T12:17:47.245234Z"}}
#Now, lets split the data.
X_train= train.iloc[:, 1:]
y_train= train['Survived'].values.reshape(-1,1)

X_test= test

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:17:48.080607Z","iopub.execute_input":"2021-07-07T12:17:48.081059Z","iopub.status.idle":"2021-07-07T12:17:48.087719Z","shell.execute_reply.started":"2021-07-07T12:17:48.081021Z","shell.execute_reply":"2021-07-07T12:17:48.085382Z"}}
#Feature Scaling is used to standardize the independent variables present in the data in a fixed range.
from sklearn.preprocessing import StandardScaler
ss= StandardScaler()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:18:00.618810Z","iopub.execute_input":"2021-07-07T12:18:00.619233Z","iopub.status.idle":"2021-07-07T12:18:00.647846Z","shell.execute_reply.started":"2021-07-07T12:18:00.619199Z","shell.execute_reply":"2021-07-07T12:18:00.646526Z"}}
features= ['Age', 'SibSp', 'Fare']

X_train[features]= ss.fit_transform(X_train[features])
X_test[features]= ss.fit_transform(X_test[features])

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:18:24.220892Z","iopub.execute_input":"2021-07-07T12:18:24.221409Z","iopub.status.idle":"2021-07-07T12:18:24.237538Z","shell.execute_reply.started":"2021-07-07T12:18:24.221369Z","shell.execute_reply":"2021-07-07T12:18:24.236531Z"}}
X_train.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:18:40.923022Z","iopub.execute_input":"2021-07-07T12:18:40.923553Z","iopub.status.idle":"2021-07-07T12:18:40.938027Z","shell.execute_reply.started":"2021-07-07T12:18:40.923517Z","shell.execute_reply":"2021-07-07T12:18:40.937034Z"}}
X_test.head()

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:19:02.778731Z","iopub.execute_input":"2021-07-07T12:19:02.779267Z","iopub.status.idle":"2021-07-07T12:19:02.806235Z","shell.execute_reply.started":"2021-07-07T12:19:02.779220Z","shell.execute_reply":"2021-07-07T12:19:02.805154Z"}}
from sklearn.linear_model import LogisticRegression

clf= LogisticRegression()

clf.fit(X_train, y_train.ravel())

predictions= clf.predict(X_test)

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:19:10.252811Z","iopub.execute_input":"2021-07-07T12:19:10.253184Z","iopub.status.idle":"2021-07-07T12:19:10.262887Z","shell.execute_reply.started":"2021-07-07T12:19:10.253151Z","shell.execute_reply":"2021-07-07T12:19:10.261218Z"}}
print(clf.score(X_train, y_train))

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:19:40.373522Z","iopub.execute_input":"2021-07-07T12:19:40.374105Z","iopub.status.idle":"2021-07-07T12:19:40.382884Z","shell.execute_reply.started":"2021-07-07T12:19:40.374070Z","shell.execute_reply":"2021-07-07T12:19:40.381330Z"}}
submission= pd.DataFrame({'PassengerId' : test_df['PassengerId'], 'Survived': predictions })

print(submission.head())

# %% [code] {"execution":{"iopub.status.busy":"2021-07-07T12:20:07.579077Z","iopub.execute_input":"2021-07-07T12:20:07.579470Z","iopub.status.idle":"2021-07-07T12:20:07.589007Z","shell.execute_reply.started":"2021-07-07T12:20:07.579440Z","shell.execute_reply":"2021-07-07T12:20:07.587497Z"}}
filename= 'titanic predictions.csv'
submission.to_csv(filename, index=False)


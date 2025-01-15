{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## predictive analytics klasifikasi pada faktor faktor hubungan yang mempengaruhi penyakit jantung"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### menggunakan algoritma KNN"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "import library dan data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.preprocessing import OneHotEncoder"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Nama dataframe kita adalah df yang berisi data dari kc_house_data.csv.\n",
    "#Features yang digunakan adalah 'bedrooms','bathrooms','sqft_living','grade','price' dan 'yr_built'\n",
    "df = pd.read_csv('heart.csv')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Sneak Peek Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Age</th>\n",
       "      <th>Sex</th>\n",
       "      <th>ChestPainType</th>\n",
       "      <th>RestingBP</th>\n",
       "      <th>Cholesterol</th>\n",
       "      <th>FastingBS</th>\n",
       "      <th>RestingECG</th>\n",
       "      <th>MaxHR</th>\n",
       "      <th>ExerciseAngina</th>\n",
       "      <th>Oldpeak</th>\n",
       "      <th>ST_Slope</th>\n",
       "      <th>HeartDisease</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>40</td>\n",
       "      <td>M</td>\n",
       "      <td>ATA</td>\n",
       "      <td>140</td>\n",
       "      <td>289</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>172</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>49</td>\n",
       "      <td>F</td>\n",
       "      <td>NAP</td>\n",
       "      <td>160</td>\n",
       "      <td>180</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>156</td>\n",
       "      <td>N</td>\n",
       "      <td>1.0</td>\n",
       "      <td>Flat</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>37</td>\n",
       "      <td>M</td>\n",
       "      <td>ATA</td>\n",
       "      <td>130</td>\n",
       "      <td>283</td>\n",
       "      <td>0</td>\n",
       "      <td>ST</td>\n",
       "      <td>98</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>48</td>\n",
       "      <td>F</td>\n",
       "      <td>ASY</td>\n",
       "      <td>138</td>\n",
       "      <td>214</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>108</td>\n",
       "      <td>Y</td>\n",
       "      <td>1.5</td>\n",
       "      <td>Flat</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>54</td>\n",
       "      <td>M</td>\n",
       "      <td>NAP</td>\n",
       "      <td>150</td>\n",
       "      <td>195</td>\n",
       "      <td>0</td>\n",
       "      <td>Normal</td>\n",
       "      <td>122</td>\n",
       "      <td>N</td>\n",
       "      <td>0.0</td>\n",
       "      <td>Up</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  MaxHR  \\\n",
       "0   40   M           ATA        140          289          0     Normal    172   \n",
       "1   49   F           NAP        160          180          0     Normal    156   \n",
       "2   37   M           ATA        130          283          0         ST     98   \n",
       "3   48   F           ASY        138          214          0     Normal    108   \n",
       "4   54   M           NAP        150          195          0     Normal    122   \n",
       "\n",
       "  ExerciseAngina  Oldpeak ST_Slope  HeartDisease  \n",
       "0              N      0.0       Up             0  \n",
       "1              N      1.0     Flat             1  \n",
       "2              N      0.0       Up             0  \n",
       "3              Y      1.5     Flat             1  \n",
       "4              N      0.0       Up             0  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Melihat 5 baris teratas dari data\n",
    "#Independent variabel(x) adalah bedrooms, bathrooms, sqft_living, grade, yr_built\n",
    "#Dependent variabel(y) adalah price\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "     Age Sex ChestPainType  RestingBP  Cholesterol  FastingBS RestingECG  \\\n",
      "913   45   M            TA        110          264          0     Normal   \n",
      "914   68   M           ASY        144          193          1     Normal   \n",
      "915   57   M           ASY        130          131          0     Normal   \n",
      "916   57   F           ATA        130          236          0        LVH   \n",
      "917   38   M           NAP        138          175          0     Normal   \n",
      "\n",
      "     MaxHR ExerciseAngina  Oldpeak ST_Slope  HeartDisease  \n",
      "913    132              N      1.2     Flat             1  \n",
      "914    141              N      3.4     Flat             1  \n",
      "915    115              Y      1.2     Flat             1  \n",
      "916    174              N      0.0     Flat             1  \n",
      "917    173              N      0.0       Up             0  \n"
     ]
    }
   ],
   "source": [
    "print(df.tail())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### penjelasan kolom"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(918, 12)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Mengetahui jumlah kolom dan baris dari data\n",
    "#Data kita mempunya 12 kolom (features) dengan 918 baris\n",
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 918 entries, 0 to 917\n",
      "Data columns (total 12 columns):\n",
      " #   Column          Non-Null Count  Dtype  \n",
      "---  ------          --------------  -----  \n",
      " 0   Age             918 non-null    int64  \n",
      " 1   Sex             918 non-null    object \n",
      " 2   ChestPainType   918 non-null    object \n",
      " 3   RestingBP       918 non-null    int64  \n",
      " 4   Cholesterol     918 non-null    int64  \n",
      " 5   FastingBS       918 non-null    int64  \n",
      " 6   RestingECG      918 non-null    object \n",
      " 7   MaxHR           918 non-null    int64  \n",
      " 8   ExerciseAngina  918 non-null    object \n",
      " 9   Oldpeak         918 non-null    float64\n",
      " 10  ST_Slope        918 non-null    object \n",
      " 11  HeartDisease    918 non-null    int64  \n",
      "dtypes: float64(1), int64(6), object(5)\n",
      "memory usage: 86.2+ KB\n"
     ]
    }
   ],
   "source": [
    "df.columns = df.columns.str.strip()\n",
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Age               0\n",
      "Sex               0\n",
      "ChestPainType     0\n",
      "RestingBP         0\n",
      "Cholesterol       0\n",
      "FastingBS         0\n",
      "RestingECG        0\n",
      "MaxHR             0\n",
      "ExerciseAngina    0\n",
      "Oldpeak           0\n",
      "ST_Slope          0\n",
      "HeartDisease      0\n",
      "dtype: int64\n"
     ]
    }
   ],
   "source": [
    "#cek apakah ada missing value \n",
    "print(df.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sex\n",
      "M    725\n",
      "F    193\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Mengubah data menjadi string dan menghitung nilai unik\n",
    "value_counts = df['Sex'].astype(str).value_counts(dropna=False)\n",
    "\n",
    "print(value_counts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "imputer = SimpleImputer(strategy='most_frequent')\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_encoder = LabelEncoder()\n",
    "df['Sex'] = label_encoder.fit_transform(df['Sex'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Sex\n",
      "1    725\n",
      "0    193\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "value_counts = df['Sex'].astype(str).value_counts(dropna=False)\n",
    "\n",
    "print(value_counts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # Inisialisasi OneHotEncoder\n",
    "# encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False agar output berupa array, bukan sparse matrix\n",
    "\n",
    "# # Terapkan OneHotEncoder\n",
    "# one_hot_encoded = encoder.fit_transform(df[['Sex']])\n",
    "\n",
    "# # Dapatkan nama kolom baru dari encoder\n",
    "# one_hot_columns = encoder.get_feature_names_out(['Sex'])\n",
    "\n",
    "# # Buat DataFrame baru dari hasil One-Hot Encoding\n",
    "# one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)\n",
    "\n",
    "# # Gabungkan dengan DataFrame asli (atau gantikan kolom asli)\n",
    "# df = pd.concat([df.drop('Sex', axis=1), one_hot_df], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "imputer = SimpleImputer(strategy='most_frequent')\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_encoder = LabelEncoder()\n",
    "df['ChestPainType'] = label_encoder.fit_transform(df['ChestPainType'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ChestPainType\n",
      "0    496\n",
      "2    203\n",
      "1    173\n",
      "3     46\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "# Mengubah data menjadi string dan menghitung nilai unik\n",
    "value_counts = df['ChestPainType'].astype(str).value_counts(dropna=False)\n",
    "\n",
    "print(value_counts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # Inisialisasi OneHotEncoder\n",
    "# encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False agar output berupa array, bukan sparse matrix\n",
    "\n",
    "# # Terapkan OneHotEncoder\n",
    "# one_hot_encoded = encoder.fit_transform(df[['ChestPainType']])\n",
    "\n",
    "# # Dapatkan nama kolom baru dari encoder\n",
    "# one_hot_columns = encoder.get_feature_names_out(['ChestPainType'])\n",
    "\n",
    "# # Buat DataFrame baru dari hasil One-Hot Encoding\n",
    "# one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)\n",
    "\n",
    "# # Gabungkan dengan DataFrame asli (atau gantikan kolom asli)\n",
    "# df = pd.concat([df.drop('ChestPainType', axis=1), one_hot_df], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "imputer = SimpleImputer(strategy='most_frequent')\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_encoder = LabelEncoder()\n",
    "df['RestingECG'] = label_encoder.fit_transform(df['RestingECG'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "RestingECG\n",
      "1    552\n",
      "0    188\n",
      "2    178\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "value_counts = df['RestingECG'].astype(str).value_counts(dropna=False)\n",
    "\n",
    "print(value_counts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # Inisialisasi OneHotEncoder\n",
    "# encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False agar output berupa array, bukan sparse matrix\n",
    "\n",
    "# # Terapkan OneHotEncoder\n",
    "# one_hot_encoded = encoder.fit_transform(df[['RestingECG']])\n",
    "\n",
    "# # Dapatkan nama kolom baru dari encoder\n",
    "# one_hot_columns = encoder.get_feature_names_out(['RestingECG'])\n",
    "\n",
    "# # Buat DataFrame baru dari hasil One-Hot Encoding\n",
    "# one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)\n",
    "\n",
    "# # Gabungkan dengan DataFrame asli (atau gantikan kolom asli)\n",
    "# df = pd.concat([df.drop('RestingECG', axis=1), one_hot_df], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "imputer = SimpleImputer(strategy='most_frequent')\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_encoder = LabelEncoder()\n",
    "df['ExerciseAngina'] = label_encoder.fit_transform(df['ExerciseAngina'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ExerciseAngina\n",
      "0    547\n",
      "1    371\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "value_counts = df['ExerciseAngina'].astype(str).value_counts(dropna=False)\n",
    "\n",
    "print(value_counts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # Inisialisasi OneHotEncoder\n",
    "# encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False agar output berupa array, bukan sparse matrix\n",
    "\n",
    "# # Terapkan OneHotEncoder\n",
    "# one_hot_encoded = encoder.fit_transform(df[['ExerciseAngina']])\n",
    "\n",
    "# # Dapatkan nama kolom baru dari encoder\n",
    "# one_hot_columns = encoder.get_feature_names_out(['ExerciseAngina'])\n",
    "\n",
    "# # Buat DataFrame baru dari hasil One-Hot Encoding\n",
    "# one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)\n",
    "\n",
    "# # Gabungkan dengan DataFrame asli (atau gantikan kolom asli)\n",
    "# df = pd.concat([df.drop('ExerciseAngina', axis=1), one_hot_df], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.impute import SimpleImputer\n",
    "imputer = SimpleImputer(strategy='most_frequent')\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "label_encoder = LabelEncoder()\n",
    "df['ST_Slope'] = label_encoder.fit_transform(df['ST_Slope'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "ST_Slope\n",
      "1    460\n",
      "2    395\n",
      "0     63\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "value_counts = df['ST_Slope'].astype(str).value_counts(dropna=False)\n",
    "\n",
    "print(value_counts)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [],
   "source": [
    "# # Inisialisasi OneHotEncoder\n",
    "# encoder = OneHotEncoder(sparse_output=False)  # sparse_output=False agar output berupa array, bukan sparse matrix\n",
    "\n",
    "# # Terapkan OneHotEncoder\n",
    "# one_hot_encoded = encoder.fit_transform(df[['ST_Slope']])\n",
    "\n",
    "# # Dapatkan nama kolom baru dari encoder\n",
    "# one_hot_columns = encoder.get_feature_names_out(['ST_Slope'])\n",
    "\n",
    "# # Buat DataFrame baru dari hasil One-Hot Encoding\n",
    "# one_hot_df = pd.DataFrame(one_hot_encoded, columns=one_hot_columns)\n",
    "\n",
    "# # Gabungkan dengan DataFrame asli (atau gantikan kolom asli)\n",
    "# df = pd.concat([df.drop('ST_Slope', axis=1), one_hot_df], axis=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "HeartDisease\n",
      "1    508\n",
      "0    410\n",
      "Name: count, dtype: int64\n"
     ]
    }
   ],
   "source": [
    "value_counts = df['HeartDisease'].astype(str).value_counts(dropna=False)\n",
    "\n",
    "print(value_counts)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### visualisasi label"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              jumlah sampel  persentase\n",
      "HeartDisease                           \n",
      "1                       508        55.3\n",
      "0                       410        44.7\n"
     ]
    }
   ],
   "source": [
    "fitur = \"HeartDisease\"\n",
    "count = df[fitur].value_counts()\n",
    "percent = 100*df[fitur].value_counts(normalize=True)\n",
    "dt = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})\n",
    "print(dt)\n",
    "count.plot(kind='bar', title=fitur);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### EDA ( Deskripsi Data )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              Age         Sex  ChestPainType   RestingBP  Cholesterol  \\\n",
      "count  918.000000  918.000000     918.000000  918.000000   918.000000   \n",
      "mean    53.510893    0.789760       0.781046  132.396514   198.799564   \n",
      "std      9.432617    0.407701       0.956519   18.514154   109.384145   \n",
      "min     28.000000    0.000000       0.000000    0.000000     0.000000   \n",
      "25%     47.000000    1.000000       0.000000  120.000000   173.250000   \n",
      "50%     54.000000    1.000000       0.000000  130.000000   223.000000   \n",
      "75%     60.000000    1.000000       2.000000  140.000000   267.000000   \n",
      "max     77.000000    1.000000       3.000000  200.000000   603.000000   \n",
      "\n",
      "        FastingBS  RestingECG       MaxHR  ExerciseAngina     Oldpeak  \\\n",
      "count  918.000000  918.000000  918.000000      918.000000  918.000000   \n",
      "mean     0.233115    0.989107  136.809368        0.404139    0.887364   \n",
      "std      0.423046    0.631671   25.460334        0.490992    1.066570   \n",
      "min      0.000000    0.000000   60.000000        0.000000   -2.600000   \n",
      "25%      0.000000    1.000000  120.000000        0.000000    0.000000   \n",
      "50%      0.000000    1.000000  138.000000        0.000000    0.600000   \n",
      "75%      0.000000    1.000000  156.000000        1.000000    1.500000   \n",
      "max      1.000000    2.000000  202.000000        1.000000    6.200000   \n",
      "\n",
      "         ST_Slope  HeartDisease  \n",
      "count  918.000000    918.000000  \n",
      "mean     1.361656      0.553377  \n",
      "std      0.607056      0.497414  \n",
      "min      0.000000      0.000000  \n",
      "25%      1.000000      0.000000  \n",
      "50%      1.000000      1.000000  \n",
      "75%      2.000000      1.000000  \n",
      "max      2.000000      1.000000  \n",
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 918 entries, 0 to 917\n",
      "Data columns (total 12 columns):\n",
      " #   Column          Non-Null Count  Dtype  \n",
      "---  ------          --------------  -----  \n",
      " 0   Age             918 non-null    int64  \n",
      " 1   Sex             918 non-null    int32  \n",
      " 2   ChestPainType   918 non-null    int32  \n",
      " 3   RestingBP       918 non-null    int64  \n",
      " 4   Cholesterol     918 non-null    int64  \n",
      " 5   FastingBS       918 non-null    int64  \n",
      " 6   RestingECG      918 non-null    int32  \n",
      " 7   MaxHR           918 non-null    int64  \n",
      " 8   ExerciseAngina  918 non-null    int32  \n",
      " 9   Oldpeak         918 non-null    float64\n",
      " 10  ST_Slope        918 non-null    int32  \n",
      " 11  HeartDisease    918 non-null    int64  \n",
      "dtypes: float64(1), int32(5), int64(6)\n",
      "memory usage: 68.3 KB\n",
      "None\n",
      "(918, 12)\n"
     ]
    }
   ],
   "source": [
    "print(df.describe())\n",
    "print(df.info())\n",
    "print(df.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Cek kembali missing value, duplikasi, inkonsisten dan Outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Age               0\n",
      "Sex               0\n",
      "ChestPainType     0\n",
      "RestingBP         0\n",
      "Cholesterol       0\n",
      "FastingBS         0\n",
      "RestingECG        0\n",
      "MaxHR             0\n",
      "ExerciseAngina    0\n",
      "Oldpeak           0\n",
      "ST_Slope          0\n",
      "HeartDisease      0\n",
      "dtype: int64\n",
      "Age               0\n",
      "Sex               0\n",
      "ChestPainType     0\n",
      "RestingBP         0\n",
      "Cholesterol       0\n",
      "FastingBS         0\n",
      "RestingECG        0\n",
      "MaxHR             0\n",
      "ExerciseAngina    0\n",
      "Oldpeak           0\n",
      "ST_Slope          0\n",
      "HeartDisease      0\n",
      "dtype: int64\n",
      "data duplikasi: 0\n"
     ]
    }
   ],
   "source": [
    "# mencari missing value\n",
    "print(df.isnull().sum())\n",
    "print(df.isna().sum())\n",
    "\n",
    "#mencari data terduplikasi\n",
    "print(\"data duplikasi:\", df.duplicated().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Age: 0\n",
      "RestingBP: 0\n",
      "Cholesterol: 0\n",
      "FastingBS: 0\n",
      "MaxHR: 0\n",
      "Oldpeak: 13\n"
     ]
    }
   ],
   "source": [
    "#mencari nilai yang tidak konsisten (nilai negatif pada data)\n",
    "continuous_features = ['Age',\n",
    "                      'RestingBP',\n",
    "                      'Cholesterol',\n",
    "                      'FastingBS',\n",
    "                      'MaxHR','Oldpeak',\n",
    "                     ]\n",
    "for feature in continuous_features:\n",
    "    print(str(feature)+': '+str(sum(df[feature] < 0)))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### visualisasi outliers dengan boxplot"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Axes(0.125,0.11;0.775x0.77)\n"
     ]
    }
   ],
   "source": [
    "#boxlplot kolom age\n",
    "print(sns.boxplot(x=df['Age']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Axes(0.125,0.11;0.775x0.77)\n"
     ]
    }
   ],
   "source": [
    "#boxlplot kolom RestingBP\n",
    "print(sns.boxplot(x=df['RestingBP']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Axes(0.125,0.11;0.775x0.77)\n"
     ]
    }
   ],
   "source": [
    "#boxlplot kolom RestingBP\n",
    "print(sns.boxplot(x=df['Cholesterol']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Axes(0.125,0.11;0.775x0.77)\n"
     ]
    }
   ],
   "source": [
    "#boxlplot kolom RestingBP\n",
    "print(sns.boxplot(x=df['MaxHR']))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Axes(0.125,0.11;0.775x0.77)\n"
     ]
    }
   ],
   "source": [
    "#boxlplot kolom RestingBP\n",
    "print(sns.boxplot(x=df['Oldpeak']))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Mengatasi Outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "# #identifikasi outlier dan ubah ke null\n",
    "# for i in ['RestingBP',\n",
    "#                       'Cholesterol',\n",
    "#                       'FastingBS',\n",
    "#                       'MaxHR','Oldpeak',\n",
    "#                      ]:\n",
    "#   Q1,Q3 = np.percentile(df.loc[:,i],[25,75])\n",
    "#   IQR = Q3 - Q1\n",
    "#   upper = Q3+(1.5*IQR)\n",
    "#   lower = Q1-(1.5*IQR)\n",
    "#   df.loc[df[i] < lower,i] = np.nan\n",
    "#   df.loc[df[i] > upper,i] = np.nan\n",
    "\n",
    "# df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "# #imputasi outlier dengan median\n",
    "# columnc=['RestingBP',\n",
    "#                       'Cholesterol',\n",
    "#                       'FastingBS',\n",
    "#                       'MaxHR','Oldpeak',]\n",
    "# for i in columnc:\n",
    "#   df.loc[df.loc[:,i].isnull(),i]=df.loc[:,i].median()\n",
    "\n",
    "# print(df.isnull().sum())\n",
    "# print(df.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\USER\\AppData\\Local\\Temp\\ipykernel_11864\\885393822.py:9: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show();\n",
      "C:\\Users\\USER\\AppData\\Local\\Temp\\ipykernel_11864\\885393822.py:9: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show();\n",
      "C:\\Users\\USER\\AppData\\Local\\Temp\\ipykernel_11864\\885393822.py:9: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show();\n",
      "C:\\Users\\USER\\AppData\\Local\\Temp\\ipykernel_11864\\885393822.py:9: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show();\n",
      "C:\\Users\\USER\\AppData\\Local\\Temp\\ipykernel_11864\\885393822.py:9: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show();\n",
      "C:\\Users\\USER\\AppData\\Local\\Temp\\ipykernel_11864\\885393822.py:9: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show();\n"
     ]
    }
   ],
   "source": [
    "#membuat boxplot setelah menangani outlier\n",
    "\n",
    "for cf in continuous_features:\n",
    "    plt.boxplot(df[cf], vert=False)\n",
    "    plt.title(cf)\n",
    "    plt.xlabel(xlabel = cf,\n",
    "               rotation=90)\n",
    "\n",
    "    plt.show();"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### EDA ( Univariate analysis )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\USER\\AppData\\Local\\Temp\\ipykernel_11864\\4090249729.py:6: UserWarning: FigureCanvasAgg is non-interactive, and thus cannot be shown\n",
      "  plt.show()\n"
     ]
    }
   ],
   "source": [
    "# univariate EDA\n",
    "numerical_features = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']\n",
    "categorical_features = ['sex', 'ChestPainType', 'FastingBS', 'ExerciseAngina', 'ST_Slope', 'HeartDisease']\n",
    "\n",
    "df.hist(bins=50, figsize=(20,15))\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              jumlah sampel  persentase\n",
      "HeartDisease                           \n",
      "1                       508        55.3\n",
      "0                       410        44.7\n"
     ]
    }
   ],
   "source": [
    "feature = categorical_features[5]\n",
    "count = df[feature].value_counts()\n",
    "percent = 100*df[feature].value_counts(normalize=True)\n",
    "dt = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})\n",
    "print(dt)\n",
    "count.plot(kind='bar', title=feature);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### EDA (Multivariate analysis)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Melihat korelasi antar variabel dengan heatmap"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(12, 10))\n",
    "sns.heatmap(df.corr(), \n",
    "            annot=True);"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### split data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "metadata": {},
   "outputs": [],
   "source": [
    "#split data 80 20\n",
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "X = df.drop([\"HeartDisease\"],axis =1)\n",
    "y = df[\"HeartDisease\"]\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Total # of sample in whole dataset: 918\n",
      "Total # of sample in train dataset: 734\n",
      "Total # of sample in test dataset: 184\n"
     ]
    }
   ],
   "source": [
    "print(f'Total # of sample in whole dataset: {len(X)}')\n",
    "print(f'Total # of sample in train dataset: {len(X_train)}')\n",
    "print(f'Total # of sample in test dataset: {len(X_test)}')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Model Development"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model Logistik Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\USER\\anaconda3\\Lib\\site-packages\\sklearn\\linear_model\\_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    }
   ],
   "source": [
    "#model Logistic Regression\n",
    "from sklearn.metrics import accuracy_score, classification_report\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "model_logreg = LogisticRegression(solver='lbfgs', multi_class='auto')\n",
    "model_logreg.fit(X_train,y_train)\n",
    "y_pred = model_logreg.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Model KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "metadata": {},
   "outputs": [],
   "source": [
    "#model KNN\n",
    "from sklearn.neighbors import KNeighborsClassifier\n",
    "knn=KNeighborsClassifier(n_neighbors=3)\n",
    "knn.fit(X_train,y_train)\n",
    "y_pred2 = knn.predict(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Evaluasi Model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### model LR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.84      0.77      0.80        77\n",
      "           1       0.84      0.90      0.87       107\n",
      "\n",
      "    accuracy                           0.84       184\n",
      "   macro avg       0.84      0.83      0.84       184\n",
      "weighted avg       0.84      0.84      0.84       184\n",
      "\n",
      "akurasi LR : 0.842391304347826\n"
     ]
    }
   ],
   "source": [
    "# Model logistik Regression\n",
    "print(classification_report(y_test, y_pred))\n",
    "print(\"akurasi LR :\", accuracy_score(y_test, y_pred))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Model KNN"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.57      0.55      0.56        77\n",
      "           1       0.68      0.70      0.69       107\n",
      "\n",
      "    accuracy                           0.64       184\n",
      "   macro avg       0.62      0.62      0.62       184\n",
      "weighted avg       0.63      0.64      0.63       184\n",
      "\n",
      "akurasi model knn: 0.6358695652173914\n"
     ]
    }
   ],
   "source": [
    "# Model  KNN\n",
    "print(classification_report(y_test, y_pred2))\n",
    "print(\"akurasi model knn:\", knn.score(X_test,y_test))"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Diketahui menggunakan Model Logistik Regresi untuk memprediksi penyakit jantung memiliki tingkat akurasi 85.71% lebih tinggi dari model KNN yang memiliki tingkat akurasi 64.13%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.to_csv('datasets-heartnew1.csv', index=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "c:\\Users\\USER\\anaconda3\\Lib\\site-packages\\sklearn\\base.py:493: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names\n",
      "  warnings.warn(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array([1], dtype=int64)"
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#Prediksi penyakit jantung dengan memasukan data menggunakan Logistik regression\n",
    "model_logreg.predict([[37,1,0,140,207,0,1,130,1,1.5,1]])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "# Streamlit App\n",
    "st.title(\"Aplikasi Prediksi Penyakit Jantung untuk Praktisi Kesehatan\")\n",
    "\n",
    "st.write(\"Aplikasi ini dirancang untuk membantu praktisi kesehatan dalam memprediksi kemungkinan seorang pasien terindikasi penyakit jantung berdasarkan data medis yang tersedia.\")\n",
    "\n",
    "st.header(\"Input Data Pasien\")\n",
    "\n",
    "age = st.number_input(\"Umur\", min_value=1, max_value=120, value=30)\n",
    "sex = st.selectbox(\"Jenis Kelamin\", [\"Laki-laki\", \"Perempuan\"])\n",
    "chest_pain = st.selectbox(\"Jenis Nyeri Dada\", [\"Angina Stabil\", \"Angina Tidak Stabil\", \"Asimptomatik\", \"Nyeri Dada Lainnya\"])\n",
    "resting_bp = st.number_input(\"Tekanan Darah Saat Istirahat (mmHg)\", min_value=50, max_value=200, value=120)\n",
    "cholesterol = st.number_input(\"Kolesterol (mg/dL)\", min_value=100, max_value=400, value=200)\n",
    "fasting_bs = st.selectbox(\"Gula Darah Puasa (> 120 mg/dL)\", [\"Ya\", \"Tidak\"])\n",
    "resting_ecg = st.selectbox(\"Hasil EKG Saat Istirahat\", [\"Normal\", \"Memiliki Kelainan Gelombang ST\", \"Memiliki Hipertrofi Ventrikel Kiri\"])\n",
    "max_hr = st.number_input(\"Detak Jantung Maksimum\", min_value=60, max_value=220, value=100)\n",
    "exercise_angina = st.selectbox(\"Angina Selama Latihan\", [\"Ya\", \"Tidak\"])\n",
    "oldpeak = st.number_input(\"Depresi ST (Oldpeak)\", min_value=0.0, max_value=10.0, value=0.0)\n",
    "st_slope = st.selectbox(\"Kemiringan Segmen ST\", [\"Meningkat\", \"Datar\", \"Menurun\"])\n",
    "\n",
    "# Mapping inputs to model format\n",
    "input_data = pd.DataFrame({\n",
    "    \"Age\": [age],\n",
    "    \"Sex\": [1 if sex == \"Laki-laki\" else 0],\n",
    "    \"ChestPainType\": [label_encoder[\"ChestPainType\"].transform([chest_pain])[0]],\n",
    "    \"RestingBP\": [resting_bp],\n",
    "    \"Cholesterol\": [cholesterol],\n",
    "    \"FastingBS\": [1 if fasting_bs == \"Ya\" else 0],\n",
    "    \"RestingECG\": [label_encoder[\"RestingECG\"].transform([resting_ecg])[0]],\n",
    "    \"MaxHR\": [max_hr],\n",
    "    \"ExerciseAngina\": [1 if exercise_angina == \"Ya\" else 0],\n",
    "    \"Oldpeak\": [oldpeak],\n",
    "    \"ST_Slope\": [label_encoder[\"ST_Slope\"].transform([st_slope])[0]]\n",
    "})\n",
    "\n",
    "# Prediction\n",
    "prediction = model_logreg.predict(input_data)[0]\n",
    "\n",
    "st.header(\"Hasil Prediksi\")\n",
    "if prediction == 1:\n",
    "    st.error(\"Pasien mungkin memiliki risiko penyakit jantung.\")\n",
    "else:\n",
    "    st.success(\"Pasien kemungkinan tidak memiliki risiko penyakit jantung yang signifikan.\")\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

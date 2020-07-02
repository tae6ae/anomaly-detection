# Anomaly Detection

## Ionosphere

| Sample | Dimension | Used Dimension | Outlier | Test Data |
|:---:|:---:|:---:|:---:|:---:|
| 351 | 35 | 35 | 126 (35.9%) | 70 (20%) |

- Data Information : Features of ionosphere to determine the existence of free electrons

- Model : Autoencoder

- Result

![3](https://user-images.githubusercontent.com/49182823/86327675-a34f3600-bc7e-11ea-9301-276d23dec447.png)

![2](https://user-images.githubusercontent.com/49182823/86327586-8581d100-bc7e-11ea-8034-e4fd1af2df9d.png)

## KDD Cup 99

| Sample | Dimension | Used Dimension | Outlier | Test Data |
|:---:|:---:|:---:|:---:|:---:|
| 494021 | 41 | 38 | 396743 (80.3%) | 98804 (21%) |

- Data Information : Connection records for detecting network intrusions

- Model : Autoencoder

- Result

![4](https://user-images.githubusercontent.com/49182823/86327791-c7127c00-bc7e-11ea-89e4-7defb4b3a526.png)

![1](https://user-images.githubusercontent.com/49182823/86326151-3e92dc00-bc7c-11ea-80fd-32085c316090.png)

## APS Failure

| Sample | Dimension | Used Dimension | Outlier | Test Data |
|:---:|:---:|:---:|:---:|:---:|
| 76000 | 170 | 170 | 1375 (1.8%) | 16000 (21%) |

- Data Information : Air Pressure System Failure in Scania Truck

- Model : Autoencoder

- Result

![5](https://user-images.githubusercontent.com/49182823/86327993-18bb0680-bc7f-11ea-9530-de7da6d1c0b0.png)

![2](https://user-images.githubusercontent.com/49182823/86248078-57ea4880-bbe8-11ea-9ed9-a738d0a38a86.png)

## DJIA Stock

| Sample | Dimension | Used Dimension | Test Data |
|:---:|:---:|:---:|:---:|
| 3019 | 7 | 2 | 755 (25%) |

- Data Information : Stock prices of 30 companies used in the Dow Jones industrial average

- Only date and market closing prices were used among data dimensions

- Model : LSTM

- Result

![3](https://user-images.githubusercontent.com/49182823/86248794-54a38c80-bbe9-11ea-937f-67629fd077c1.png)

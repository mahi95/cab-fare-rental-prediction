# cab fare prediciton 

#removing all the objects
rm(list = ls())

#set the working directory
setwd("C:/Mahaanand/Machine Learning/edwisor_machine_learning/my_projects/cab_rental")

# Loading required libraries for the project
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced","C50",
      "dummies", "e1071", "Information", "MASS", "rpart", "gbm", "ROSE",
      'sampling', 'DataCombine', 'inTrees',"scales","psych","gplots")

# installing the packages
#install.packages(x)

# checking all the libraries are installed successfully or not
lapply(x, require, character.only = TRUE)
rm(x)

# Loading the data and data exploration
master_df = read.csv('train_cab.csv')

# check the dimensions of the data
dim(master_df)

# check the structure of the data
str(master_df)

# check the class of the data
class(master_df)

# checking the head of the data
head(master_df, 8)

# checking the tail of the data
tail(master_df, 8)

# summary of the data
summary(master_df)

# structure of the fare amount
str(master_df$fare_amount)

# converting the fare amount into numeric
master_df$fare_amount = as.numeric(as.character(master_df$fare_amount))

# summary of fare_amount
summary(master_df$fare_amount)

################################# Missing value analysis #################################
# Missing value analysis
sum(is.na(master_df))

# finding missing values in each column
missing_df = data.frame(apply(master_df, 2, function(x) { sum(is.na(x)) }))

# creating a variable column
missing_df$variable_column = row.names(missing_df)

#changing the name of the first variable
names(missing_df)[1] = 'missing_val_count'

# calculating missing_val_percentage
missing_df$missing_perc = (missing_df$missing_val_count / nrow(master_df)) * 100

#sorting the values in ascending order
missing_df = missing_df[order(-missing_df$missing_perc),]

#rearranging the columns
missing_df = missing_df[, c(2,1,3)]

# eliminating the missing values
master_df = na.omit(master_df)

# re-check for the missing value analysis
sum(is.na(master_df))

################################# Outlier analysis #################################
# Box plot analysis for detecting outliers
c_index = sapply(master_df, is.numeric)
c_data = master_df[,c_index]
cnames = colnames(c_data)

# Plotting box plot
for (i in 1:length(cnames)){
  assign(paste0("gn",i), ggplot(aes_string(y=cnames[i]) , data=master_df) + stat_boxplot(geom='errorbar', width=0.5) + 
                                  geom_boxplot(outlier.colour='red', fill='grey', outlier.shape=16, outlier.size=1, notch=FALSE) +
                                  theme(legend.position="bottom")+
                                  labs(y=cnames[i])+
                                  ggtitle(paste("Outlier analysis for ", cnames[i])))
}

gridExtra::grid.arrange(gn1, gn2, gn3, gn4, gn5, gn6, ncol=2)

# removing outliers
find_boundaries = function (i){
  q1 = quantile(i, 0.25)
  q3 = quantile(i, 0.75)
  upp_lim = q3 + (1.5 * IQR(i))
  low_lim = q1 - (1.5 * IQR(i))
  return(list(upp_lim, low_lim))
}

## Fare amount ##
limits = find_boundaries(master_df$fare_amount)
fr_upp_lim = limits[[1]]
fr_low_lim = limits[[2]]

# capping the fare amount with the lower and upper limits
master_df[master_df$fare_amount < fr_low_lim, 'fare_amount'] = fr_low_lim
master_df[master_df$fare_amount > fr_upp_lim, 'fare_amount'] = fr_upp_lim

# dropping the values less than or equal to 0.01
master_df[master_df$fare_amount <=0.01, 'fare_amount'] = NA
sum(is.na(master_df))
master_df = na.omit(master_df)

colnames(master_df)

## pickup_longitude ##
pl_limits = find_boundaries(master_df$pickup_longitude)
pl_upp_lim = pl_limits[[1]]
pl_low_lim = pl_limits[[2]]

master_df[master_df$pickup_longitude<pl_low_lim, 'pickup_longitude'] = pl_low_lim
master_df[master_df$pickup_longitude>pl_upp_lim, 'pickup_longitude'] = pl_upp_lim

## pickup_latitude ##
plat_limits = find_boundaries(master_df$pickup_latitude)
plat_upp_lim = plat_limits[[1]]
plat_low_lim = plat_limits[[2]]

master_df[master_df$pickup_latitude < plat_low_lim, 'pickup_latitude'] = plat_low_lim
master_df[master_df$pickup_latitude > plat_upp_lim, 'pickup_latitude'] = plat_upp_lim

## dropoff_longitude ##
dlon_limits = find_boundaries(master_df$dropoff_longitude)
dlon_upp_lim = dlon_limits[[1]]
dlon_low_lim = dlon_limits[[2]]

master_df[master_df$dropoff_longitude < dlon_low_lim, 'dropoff_longitude'] = dlon_low_lim
master_df[master_df$dropoff_longitude > dlon_upp_lim, 'dropoff_longitude'] = dlon_upp_lim

## dropoff_latitude ##
dlat_limits = find_boundaries(master_df$dropoff_latitude)
dlat_upp_lim = dlat_limits[[1]]
dlat_low_lim = dlat_limits[[2]]

master_df[master_df$dropoff_latitude < dlat_low_lim, 'dropoff_latitude'] = dlat_low_lim
master_df[master_df$dropoff_latitude > dlat_upp_lim, 'dropoff_latitude'] = dlat_upp_lim

## Passenger count ##
pc_limits = find_boundaries(master_df$passenger_count)
pc_upp_lim = round(pc_limits[[1]])
pc_low_lim = round(pc_limits[[2]])

master_df[master_df$passenger_count < pc_low_lim, 'passenger_count'] = pc_low_lim
master_df[master_df$passenger_count > 6, 'passenger_count'] = pc_upp_lim

# removing the values less than 1 in passenger_count
master_df[master_df$passenger_count < 1, 'passenger_count'] = NA
master_df = na.omit(master_df)

# visualizing the box plot again
for (i in 1:length(cnames)){
  assign(paste0("gn",i), ggplot(aes_string(y=cnames[i]) , data=master_df) + stat_boxplot(geom='errorbar', width=0.5) + 
           geom_boxplot(outlier.colour='red', fill='grey', outlier.shape=16, outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i])+
           ggtitle(paste("Outlier analysis for ", cnames[i])))
}

gridExtra::grid.arrange(gn1, gn2, gn3, gn4, gn5, gn6, ncol=2)


################################# Feature Engineering #################################
master_df$pickup_datetime = gsub('// UTC', '', master_df$pickup_datetime)

# converting into a date
master_df$date_of_travel = as.Date(master_df$pickup_datetime)

# deriving year column
master_df$year_of_travel = substr(as.character(master_df$date_of_travel), 1, 4)

#deriving month column
master_df$month_of_travel = substr(as.character(master_df$date_of_travel), 6, 7)

# deriving day column
master_df$day_of_travel = weekdays(as.POSIXct(master_df$date_of_travel), abbreviate = FALSE)

# deriving date column
master_df$date_of_travel = substr(as.character(master_df$date_of_travel), 9, 10)

# dropping the pickup_datetime column since we have derived the necessary features from the data
master_df$pickup_datetime = NULL

# calculating distance using haversine formula
calc_geo_dist = 























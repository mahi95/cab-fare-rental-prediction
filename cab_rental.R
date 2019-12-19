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

# deriving hour column
master_df$hour_of_travel = substr(as.factor(master_df$pickup_datetime), 12, 13)

# dropping the pickup_datetime column since we have derived the necessary features from the data
master_df$pickup_datetime = NULL

# calculating distance using haversine formula
# importing the required library
library(purrr)
library(geosphere)
library(rlist)

calc_geo_dist = function(pickup_longitude, pickup_latitude, dropoff_longitude, dropoff_latitude) {
  loadNamespace('purrr')
  loadNamespace('geosphere')
  pickup_coor = purrr::map2(pickup_longitude, pickup_latitude, function(x,y) c(x,y))
  dropoff_coor = purrr::map2(dropoff_longitude, dropoff_latitude, function(x,y) c(x,y))
  distance_list = purrr::map2(pickup_coor, dropoff_coor, function(x,y) geosphere::distHaversine(x,y))
  distance = list.extract(distance_list, position=1)
  return(distance/1000.0)
}

for (i in 1:nrow(master_df)){
  master_df$distance[i] = calc_geo_dist(master_df$pickup_longitude[i], 
                                        master_df$pickup_latitude[i],
                                        master_df$dropoff_longitude[i],
                                        master_df$dropoff_latitude[i])
}

# checking for outliers in distance
ggplot(aes_string(x='distance', y='fare_amount'), data=master_df)+
  stat_boxplot(geom = 'errorbar', width=0.5) +
  geom_boxplot(oulier.colour="red", fill="grey", outlier.size = 1, notch = F) +
  theme(legend.position = "bottom") +
  ggtitle(paste("Outlier detection for distance"))

# checking the head of distance
head(master_df)

# checking the summary of distance
summary(master_df$distance)

# checking for distance less than 1(For log transformation in the future case)
dim(master_df[master_df$distance < 1,])

# Total of 2978 rows with distance less than 1
# computing them with the mean
master_df[master_df$distance < 1, 'distance'] = mean(master_df$distance)

# recheck the distance
dim(master_df[master_df$distance < 1,])

### Data preprocessing is complete. copying into the duplicate data
master_comp_preprocessed_data = master_df
dim(master_df)

################################# Exploratory data analysis #################################
### Univariate analysis ###
# appending distance with cnames
append(cnames, 'distance', after=length(cnames))

# creating a function for histogram plot for univariate analysis
create_hist_plot = function (x) {
  ggplot(master_df, aes_string(x=x)) +
    geom_histogram(fill='blue', colour='black')+
    geom_density()+
    theme_bw()+
    xlab(x) + ylab('frequency') + ggtitle(paste0('distribution of data ', x))
}

# distribution of fare_amount
create_hist_plot('fare_amount')

# distribution of pickup_longitude
create_hist_plot('pickup_longitude')

# distribution of pickup_latitude
create_hist_plot('pickup_latitude')

# distribution of dropoff_longitude
create_hist_plot('dropoff_longitude')

# distribution of dropoff_latitude
create_hist_plot('dropoff_latitude')

# distribution of passenger_count
create_hist_plot('passenger_count')

# distribution of distance
create_hist_plot('distance')

### Inferences ###
# distance data seems skewed a little bit left

### Bivariate analysis ###
# analysis between target variable and independent variable
# creating bar plot for analysis b/w categorical and target variable

create_bar_plot = function(x){
  ggplot(master_df, aes_string(x=x, y='fare_amount')) +
    geom_bar(stat='identity', color='blue')+
    labs(title=paste0('fare amount vs ', x), x=x, y='fare_amount') 
}

# bar plot b/w fare_amount and year_of_travel
create_bar_plot('year_of_travel')

# bar plot b/w fare_amount and month_of_travel
create_bar_plot('month_of_travel')

# removing nan values in month_of_travel
sum(is.na(master_df$month_of_travel))
master_df = na.omit(master_df)

# bar plot b/w fare_amount and day_of_travel
create_bar_plot('day_of_travel')

# bar plot b/w hour_of_travel and fare_amount
create_bar_plot('hour_of_travel')

# creating scatter plot for bivariate analysis b/w continuous and target variable
create_scatter_plot = function(x){
  ggplot(master_df, aes_string(x=x, y='fare_amount'))+
    geom_point(color='blue') +
    labs(title=paste0('fare_amount vs ', x), x=x, y='fare_amount')
}

# pickup_longitude
create_scatter_plot('pickup_longitude')

# pickup_latitude
create_scatter_plot('pickup_latitude')

# dropoff_longitude
create_scatter_plot('dropoff_longitude')

# dropoff_latitude
create_scatter_plot('dropoff_latitude')

# passenger_count
create_scatter_plot('passenger_count')

# distance
create_scatter_plot('distance')


################################# Feature selection #################################
# finding the continuous variables for correlation plot
c_index = sapply(master_df, is.numeric)
c_data = master_df[, c_index]
cnames = colnames(c_data)

# correlation plot for selecting features needed for model development
cor_analysis = cor(master_df[, c_index])
corrgram(master_df[,c_index], order=F, upper.panel=panel.pie, text.panel=panel.txt, main="Correlation Plot")

#Inferences
#distance is highly positively correlated with fare_amount,,, needed variable for model development

# chi-square test for categorical variables
chr_index = sapply(master_df, is.character)
chr_data = master_df[,chr_index]

for(i in 1:5){
  print(names(chr_data)[i])
  print(chisq.test(table(master_df$fare_amount, chr_data[,i])))
}


# Infernces
# p-value of date_of_travel and day_of_travel are greater than 0.05, hence dropping the columns

master_df$date_of_travel = NULL
master_df$day_of_travel = NULL


################################# Feature scaling #################################

# plotting distance to check the normality
hist(master_df$distance, col='blue')

# since the data is right skewed taking log transformation of the data
master_df$log_distance = log(master_df$distance)

# rechecking the log_distance for normality
hist(master_df$log_distance, col='blue')

## Data is normally distributed and ready for model development



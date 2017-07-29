#Incubator Challenge

setwd("/home/chasedehan/Dropbox/R Code/ML/Data Incubator")

#load libraries
library(tidyverse)
library(lubridate)

#Importing data
months <- rep(1:9)
file <- "-citibike-tripdata.csv"
read_files <- paste("20150",months,file, sep ="")
read_files <- c(read_files,paste("2015",c(10,11,12), file, sep=""))
data <- data.frame()
for(i in 1:12){  data <- rbind(data, read_csv(read_files[i])) }

#convert dates to lubridate class for easier working
data <- data %>%
  mutate(starttime = parse_date_time(starttime, c("mdy_hms", "mdy_hm")), 
         stoptime = parse_date_time(stoptime, c("mdy_hms", "mdy_hm")),
         month = month(starttime),
         hour = hour(starttime),
         duration = stoptime - starttime)  #calculated off time stamps

#median trip duration in seconds
median(data$tripduration[data$tripduration >=0]) #629
    #Was only checking to make sure greater than zero b/c my calculations from time stamps show (-) values

#Fraction of rides start and end at same location
nrow(data[data$`start station id` == data$`end station id`,]) / nrow(data) #0.02235839

#sd() of number of stations
a <- data %>%
  select(bikeid, `start station id`, `end station id`) %>%
  unique()
id <- c(a$bikeid, a$bikeid)
station_id <- unlist(c(a[,2], a[,3]))
a <- as.data.frame(cbind(id, station_id))
rownames(a) <- NULL
a <- a %>%
  unique() %>%
  group_by(id) %>%
  summarise(count = n()) %>%
  ungroup()
sd(a$count) #sd of stations visited 54.54511 

#Average duration of trips for each month in the year
b <- data %>% 
  select(month, tripduration) %>%
  group_by(month) %>%
  summarise(avg_dur = mean(tripduration))
max(b$avg_dur) - min(b$avg_dur) # 430.5703

#hourly_usage_fraction
c <- data %>%
  group_by(`start station id`, hour) %>%
  summarise(count = n())
d <- data %>%
  group_by(`start station id`) %>%
  summarise(total = n()) %>%
  merge(c, by = 1) %>%
  mutate(ratio = count / total) %>%
  arrange(desc(ratio))
d$ratio[1]  #0.388231

#By subscriber/ customer, fraction exceed limit  # 0.03810678
sub <- data %>%
  filter(usertype == "Subscriber") 
cust <- data %>%
  filter(usertype == "Customer") 
(nrow(cust[cust$tripduration > (30*60),]) + nrow(sub[sub$tripduration > (45*60),]) )/ nrow(data)

#bikes moved
moved <- data %>%
  group_by(bikeid) %>%
  arrange(starttime) %>% 
  mutate(moved = ifelse(lag(`end station id`) == `start station id`, 0, 1)) %>% 
  summarise(num_moved = sum(na.omit(moved))) #NA if first time seen in data
mean(moved$num_moved) #65.42491


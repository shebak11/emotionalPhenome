library(tidyverse)

# TODO: Change to local environment
setwd("C:/Users/Scotty/Documents/GitHub/emotionalPhenome/")  
bnet.data <- read_csv("data/BRAINnet_data_complete_outliers_removed.csv")

# Just select the Anhedonia, Anxious Arousal, and Tension factors for
# each of the 420 subjects that have that data available
bnet.dass.dat <- bnet.data %>% 
  select(starts_with("Factor3")) %>% 
  drop_na %>%
  rename(Anhedonia = Factor3_Anhedonia_PD_PTSD_MD_Allpts) %>%
  rename(AnxArousal = Factor3_AnxArousal_PD_PTSD_MD_Allpts) %>%
  rename(Tension = Factor3_Irritabliity_PD_PTSD_MD_Allpts)

# Run Hierarchical clustering with Ward's variance criterion
# to recover the original cluster solution
clust.result <- hclust(dist(bnet.dass.dat), method="ward.D2")
memb <- cutree(clust.result, k=6)  # Cut the dendrogram at 6 clusters
bnet.dass.dat <- cbind(bnet.dass.dat, clust_id = memb)  # Record the cluster labels

# Report the cluster centroids for each cluster...
bnet.dass.dat %>% 
  group_by(clust_id) %>% 
  summarise(anhedonia = mean(Anhedonia),
            anx_arousal = mean(AnxArousal),
            tension = mean(Tension))

# PUBLISHED CLUSTER CENTROIDS (Table 1) #
# - # - # - # - # - # - # - # 
# Type Cluster: (Anhedonia, Anxious arousal, Tension)
# - # - # - # - # - # - # - #
# Normative Mood: (-0.411, -0.026, -0.662)
# Tension: (-0.047, -0.504, 0.468)
# Anxious Arousal: (0.69, 1.842, 0.122)
# General Anxiety: (-0.824, 0.929, 1.201)
# Anhedonia: (2.113, -0.154, -0.958)
# Melancholia: (1.185, -1.217, 1.533)

# Let's try and match clusters to labels 
# (you may have to adjust for your local environment/machine - 
# if you do need to make adjustments, just change the "levels"
# argument to reflect the appropriate order. The "labels" argument
# has been built to match Table 1 from the paper
bnet.dass.dat$clust_label <- factor(bnet.dass.dat$clust_id, 
                                   levels=c(1, 2, 4, 3, 6, 5), 
                                   labels=c("NormativeMood",
                                            "Tension",
                                            "AnxiousArousal",
                                            "GeneralAnxiety",
                                            "Anhedonia",
                                            "Melancholia"),
                                   ordered=TRUE)

# Get the centroids for each of our clusters
cluster.centroids <- bnet.dass.dat %>% 
  group_by(clust_label) %>% 
  summarise(anhedonia = mean(Anhedonia),
            anx_arousal = mean(AnxArousal),
            tension = mean(Tension))

# Print out Table 1...
t(cluster.centroids)

my.data <- read_csv("data/my_BRAINnet_scores.csv")
my.dass.dat <- my.data %>% select(Anhedonia, AnxArousal, Tension)

library(caret)
set.seed(42)
train.indx <- createDataPartition(y=bnet.dass.dat$clust_label,
                                  p=0.7,
                                  list=FALSE)
bnet.train <- bnet.dass.dat[train.indx, ]
bnet.val <- bnet.dass.dat[-train.indx, ]

trctrl <- trainControl(method="repeatedcv",
                       number=10,
                       repeats=3)

knn.fit <- train(clust_label ~ Anhedonia + AnxArousal + Tension,
                 data=bnet.train,
                 method="knn",
                 trControl=trctrl,
                 preProcess=NULL,
                 tuneLength=10)
knn.fit
plot(knn.fit)

# Measure our KNN Cluster label classifier on 
# a held-out validation set (not surprisingly, it does 
# quite well, at accuracy = 0.9516)
val.preds <- predict(knn.fit, 
                     newdata=bnet.val)
confusionMatrix(val.preds, bnet.val$clust_label)

my.clust.labels <- predict(knn.fit, 
                           my.dass.dat)

my.data <- cbind(my.data, dass_clust_labels = my.clust.labels)
write.csv(my.data, "my_BRAINnet_data_with_cluster_labels.csv", row.names=FALSE)

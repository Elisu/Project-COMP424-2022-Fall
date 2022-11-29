rm(list = ls())
library(ggplot2)
library(data.table)

setwd("~/Documents/Uni/Study abroad/NiendeSemester/COMP 424/Project/Project-COMP424-2022-Fall")


time_b6 <- as.array(read.delim("times_player1_boardsize_6.txt")[,1])
time_b7 <- as.array(read.delim("times_player1_boardsize_7.txt")[,1])
time_b8 <- as.array(read.delim("times_player1_boardsize_8.txt")[,1])
time_b9 <- as.array(read.delim("times_player1_boardsize_9.txt")[,1])
time_b10 <- as.array(read.delim("times_player1_boardsize_10.txt")[,1])
time_b11 <- as.array(read.delim("times_player1_boardsize_11.txt")[,1])
time_b12 <- as.array(read.delim("times_player1_boardsize_12.txt")[,1])

data <- data.table(time = c(time_b6,time_b7,time_b8,time_b9,time_b10,time_b11,time_b12), 
                   board_size = c(rep(6,length(time_b6)),
                                  rep(7,length(time_b7)),
                                  rep(8,length(time_b8)),
                                  rep(9,length(time_b9)),
                                  rep(10,length(time_b10)),
                                  rep(11,length(time_b11)),
                                  rep(12,length(time_b12))),
                   round = c(seq(1,length(time_b6)),
                             seq(1,length(time_b7)),
                             seq(1,length(time_b8)),
                             seq(1,length(time_b9)),
                             seq(1,length(time_b10)),
                             seq(1,length(time_b11)),
                             seq(1,length(time_b12))))

data[,board_size := factor(as.character(board_size),as.character(seq(6,12)))]

ggplot(data, aes(x=round, y=time, group = board_size, color = board_size)) +
  geom_line()+
  theme(text = element_text(size = 20)) +
  ggtitle("Time spend on each move") + 
  xlab("Round") + 
  ylab("Time (s)") + 
  labs(color='Board size')

ggsave("times_autoplay4.eps", plot = last_plot(), width = 17, height = 12,units = "cm")
# Autoplay 2: uses median removes 1/2
# Autoplay 3: removes 2/3
# Autoplay 4: removes 3/4

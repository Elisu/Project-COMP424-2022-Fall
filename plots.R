rm(list = ls())
library(ggplot2)
library(data.table)

setwd("~/Documents/Uni/Study abroad/NiendeSemester/COMP 424/Project/Project-COMP424-2022-Fall")


time_b6 <- as.array(read.delim("data/times_player1_boardsize_6.txt")[,1])
time_b7 <- as.array(read.delim("data/times_player1_boardsize_7.txt")[,1])
time_b8 <- as.array(read.delim("data/times_player1_boardsize_8.txt")[,1])
time_b9 <- as.array(read.delim("data/times_player1_boardsize_9.txt")[,1])
time_b10 <- as.array(read.delim("data/times_player1_boardsize_10.txt")[,1])
time_b11 <- as.array(read.delim("data/times_player1_boardsize_11.txt")[,1])
time_b12 <- as.array(read.delim("data/times_player1_boardsize_12.txt")[,1])

thread_b6 <- as.array(read.delim("data/threads_player1_boardsize_6.txt")[,1])
thread_b7 <- as.array(read.delim("data/threads_player1_boardsize_7.txt")[,1])
thread_b8 <- as.array(read.delim("data/threads_player1_boardsize_8.txt")[,1])
thread_b9 <- as.array(read.delim("data/threads_player1_boardsize_9.txt")[,1])
thread_b10 <- as.array(read.delim("data/threads_player1_boardsize_10.txt")[,1])
thread_b11 <- as.array(read.delim("data/threads_player1_boardsize_11.txt")[,1])
thread_b12 <- as.array(read.delim("data/threads_player1_boardsize_12.txt")[,1])

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

data[,board_size := factor(as.character(board_size),as.character(seq(12,6)))]

data_threads <- data.table(time = c(thread_b6,thread_b7,thread_b8,thread_b9,thread_b10,thread_b11,thread_b12), 
                   board_size = c(rep(6,length(thread_b6)),
                                  rep(7,length(thread_b7)),
                                  rep(8,length(thread_b8)),
                                  rep(9,length(thread_b9)),
                                  rep(10,length(thread_b10)),
                                  rep(11,length(thread_b11)),
                                  rep(12,length(thread_b12))),
                   round = c(seq(1,length(thread_b6)),
                             seq(1,length(thread_b7)),
                             seq(1,length(thread_b8)),
                             seq(1,length(thread_b9)),
                             seq(1,length(thread_b10)),
                             seq(1,length(thread_b11)),
                             seq(1,length(thread_b12))))
data_threads[,board_size := factor(as.character(board_size),as.character(seq(12,6)))]


ggplot(data, aes(x=round, y=time, group = board_size, color = board_size)) +
  geom_line()+
  theme(text = element_text(size = 18)) +
  ggtitle("Time spent on each move") + 
  xlab("Round") + 
  ylab("Time (s)") + 
  geom_hline(yintercept=2, linetype="dashed", color = "black") +
  geom_hline(yintercept=30, linetype="dashed", color = "black") +
  labs(color='Board size')

ggsave("times_autoplay2_tuned.eps", plot = last_plot(), width = 18, height = 12,units = "cm")

ggplot(data_threads, aes(x=round, y=time, group = board_size, color = board_size)) +
  geom_line()+
  theme(text = element_text(size = 18)) +
  ggtitle("Number of simulations per move") + 
  xlab("Round") + 
  ylab("Simulations (#)") + 
  labs(color='Board size')

ggsave("threads_autoplay2_tuned.eps", plot = last_plot(), width = 18, height = 12,units = "cm")



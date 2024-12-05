# Addition Task

# Input:

For one batch sample:
$ \left(
\begin{array}{}
     {\color{red} 0.0} &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     {\color{red} 1.0} &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     {\color{red} 1.0} &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     {\color{blue} 9.0} &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     {\color{red} 1.0} &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     {\color{red} 1.0} &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     {\color{red} 0.0} &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0
\end{array}
\right) $
sequences in red

Structure:
$ b \times d \times i$
with b=batch_size, i,d=2*sequence_max_length+1

# Output:
$  \left(
\begin{array}{}
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     {\color{red} 1.0}\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     {\color{red} 0.0}\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     {\color{red} 0.0}\\
     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &     0.0 &  0.0  &     {\color{red} 1.0}
\end{array}
\right) $
result in red

# Variables:

  **batch_size = 1**
  batch size
  **sequence_length = 3**
  current sequence length
  **sequence_max_length = 5**
  maximal sequence length
  **iterations = 500000**
  number of epochs
  **summarize_freq = 100**
   when to summarize the changes
  **check_freq = 100**
  take a checkpoint every n=100 steps
  **curriculum_freq = 2500**
  increment the curriculum e.g. increase the sequence length every 2500 epochs

  **mem_slot = 32**
  number of memory slots
  **mem_size = 1**
  depth of each memory slot
  **read_heads = 2**
  number of read heads
  **curriculum_increment = 1**
  increment curriculum by 1
  **input_size = 2*sequence_max_length + 1**

  **output_size = 64**
  number of output neurons
  **replaceWithWrong = True**
  respresent the results with the highest loss
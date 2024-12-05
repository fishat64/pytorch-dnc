# Multicopy Task

# Input:

For one batch sample:
$ [ \underbrace{1, 0, 0, 1,}_{\text{sample to copy}} 9, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, \underbrace{2}_{\text{number of copies}}] $

Structure:
$ b \times d \times i$
with b=batch_size, i=number of parallel inputs (1, sequential) and d= input_length*maxnumberofcopies

# Output:
$ [\underbrace{1, 0, 0, 1,}_{\text{copy of sequence}} \underbrace{1, 0, 0, 1,}_{\text{copy of sequence}} 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] $

# Variables:

  **batch_size = 100**
  batch size
  **sequence_max_length = 3**
  beginning sequence length, in the example it's 4
  **iterations = 20000**
  maximum iterations
  **summarize_freq = 100**
  when to summarize the changes
  **check_freq = 100**
  take a checkpoint every n=100 steps
  **curriculum_freq = 5000**
  increment the curriculum e.g. increase the sequence length
  **curriculumMaxNoCopies_freq = 1000**
  increment the maximal number of copies
  **mem_slot = 16**
  how many memory slots
  **mem_size = 1**
  how big is each memory slot
  **read_heads = 1**
  how many read heads for input
  **curriculum_increment = 1**
  increase the length of the sequence by curriculum_increment if epoch%curriculum_freq = 0
  **maxnumberofcopies=6**
  maximal number of possible copies
  **currentmaxnocopies=3**
  minimal numbers of copies to begin with
  **input_length = 6**
  sequence length -> input_length*maxnumberofcopies 
  **input_size = 1**
  sequential input =1
  **output_size = 64**
  output size
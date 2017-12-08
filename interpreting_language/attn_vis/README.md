This part of the code contains Jason Z Li's contribution, a visualization of the attention weights learned for different examples. It consists of an API that contains JSON containing a list of sentence objects, each containing a text array of strings corresponding to words and a weights array of floats bw 0 and 1 corresponding to those words, with both arrays being the same length:

INPUT:
[{"text":[(str) word1, (str) word2, ...], 
  "weights" :[(float) weight1, (float) weight2...},

 {"text":[(str) word1, (str) word2, ...], 
  "weights" :[(float) weight1, (float) weight2...]},

  ...]

OUTPUT:

An HTML file containing all of the sentences where each word is highlighted more or less darkly based on what the corresponding weights.

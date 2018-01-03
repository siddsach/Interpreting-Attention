#!/usr/bin/env python

import argparse

parser = argparse.ArgumentParser(description='TTIC SLURM sbatch script creator')
parser.add_argument('INPUT_FILE', help='Input file with list of commands to run')
parser.add_argument('PARTITION', help='Name of partition to use')
parser.add_argument('-C', '--constraint', help='Constraint to use')
parser.add_argument('-J', '--job-name', help='Name of the job')

args = parser.parse_args()

def gen_sbatch_end(constraint, job_name):
  if constraint and job_name:
    sbatch_end = ' --constraint=' + args.constraint + ' --job-name=' + args.job_name
  elif constraint:
    sbatch_end = ' --constraint=' + args.constraint
  elif job_name:
    sbatch_end = ' --job-name=' + args.job_name
  else:
    sbatch_end = ''
  return sbatch_end

file_in = open(args.INPUT_FILE, 'r')
lines = file_in.readlines()

count = 0
commands = []
while count < len(lines):
  if count % 5000 == 0 and count > 0:
    index = count / 5000
    file_out = open('batch-commands-' + str(index) + '.txt', 'w')
    for i in commands:
      file_out.write(i.strip() + '\n')
    file_out.close()
    file_out = open('sbatch-script-' + str(index) + '.txt', 'w')
    file_out.write('#!/bin/bash\n')
    sbatch_end = gen_sbatch_end(args.constraint, args.job_name)
    file_out.write('#SBATCH --partition=' + args.PARTITION + ' --cpus-per-task=1 --array=1-' + str(len(commands)) + sbatch_end + '\n')
    file_out.write('bash -c "`sed "${SLURM_ARRAY_TASK_ID}q;d" '+'batch-commands-'+str(index)+'.txt'+'`"')
    file_out.close()
    commands = []
  commands.append(lines[count])
  count += 1

file_out = open('batch-commands-last.txt', 'w')
for i in commands:
  file_out.write(i.strip() + '\n')
file_out.close()
file_out = open('sbatch-script-last.txt', 'w')
file_out.write('#!/bin/bash\n')
sbatch_end = gen_sbatch_end(args.constraint, args.job_name)
file_out.write('#SBATCH --partition=' + args.PARTITION + ' --cpus-per-task=1 --array=1-' + str(len(commands)) + sbatch_end + '\n')
file_out.write('bash -c "`sed "${SLURM_ARRAY_TASK_ID}q;d" '+'batch-commands-last.txt'+'`"')
file_out.close()

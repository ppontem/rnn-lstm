import itertools
import os
import numpy as np
import collections



def replace(infile, outfile, replacements):
	for line in infile:
        	for src, target in replacements.iteritems():
                	line = line.replace(src, target)
                outfile.write(line)


def slurm_generator(slurm_temp,params_list,fname,command):

	for params in itertools.product(*params_list):
		#print L, om , V
		#params= [ri,rs,nt,cv]
		params=list(params)
		print params
		#infile=open('slurm.phase_diag_pratio.sh')
		infile = open(slurm_temp)
		#filename="scrslurm__z_finalfu_nd_ri{0}_rs{1}_nt{2}_cv{3}.sh".format(*params)
		filename=fname.format(*params)
		outfile=open("scripts/"+filename, 'w')
		replacements = {'jobname':filename,'CDIR':os.getcwd(),
				'command_s':command.format(*params) }

		replace(infile,outfile,replacements)
		infile.close()
		outfile.close()
		os.system("sbatch scripts/"+filename) #submit to slurm
		print 'Submitted ' + filename


params = collections.OrderedDict()
for seed in range(11, 12+1):
	params = collections.OrderedDict()
	np.random.seed(seed)
	params['bs'] = np.random.randint(50, 500, 1).tolist() #range(9, 9+1)
	params['hu'] = [50]
	params['lr'] = (10**np.random.uniform(-6, -2, 1)).tolist()
	params['rho'] = np.random.choice([0.9, 0.95, 0.99, 0.999], 1).tolist() #10**np.random.uniform(0.9, 0.999)[0.9, 0.99, 0.999]
	params['clip'] = [5.0]
	params_list = params.values()
	job_name_temp = "monk_lstm_bs_{}_hu_{}_lr_{}_rho_{}_clip_{}_gpp_1"
	# command = sqsub -o foo.log -r 5h ./foo arg1 arg2...
	#v2: changed eigensolver from numpy.eig to scipy.linalg.eig to see if it corrects error in convergence of eigenvalues in models.py
	#v3: forgot to change eigensolver of hamiltonians in models.py
	for params in itertools.product(*params_list):
		job_name = job_name_temp.format(*params)
		file_name = "output/" + job_name
		sharcnet_command = "sqsub -v -q gpu  --gpp=1 -o {} -r 4d ".format(file_name)
		code_command = "python mnist-lstm.py {0} {1} {2} {3} {4}".format(*params)
		command = sharcnet_command + code_command
		print command
		os.system(command)

# clusters_size_LAZARIDES(L, model_params, data_folder, seed_min, seed_max) # the goal
# slurm_generator('slurm.sh', params_list, job_name,
# 	"python clusters_LAZARIDES.py {0} {1} {2} {3} {4}")

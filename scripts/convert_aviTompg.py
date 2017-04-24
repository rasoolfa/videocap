########################################
#  This code builds clips from a video given start and end frames
#  In this code. FFMPEG has been used, so ffmpeg need to be installed 
########################################

from __future__ import print_function
import subprocess
import os
import json
import argparse
from  utils import get_list_files, write_info
from time import sleep

def make_clips(avi_files, verbose, video_dir, dest_folder):

	c_error = 0
	c_ok = 0
	l_error = []
	l_ok = []
	len_total =  len(avi_files)
	
	for vid_f in avi_files:

		new_fname = os.path.splitext(vid_f)[0] + '.mp4' ## no file extention and add mp4 extention
		command= "ffmpeg -i " + os.path.join(video_dir, vid_f)+ " -vcodec libx264 -crf 18  -c:a aac -strict -2 -pix_fmt yuv420p " + os.path.join(dest_folder, new_fname)
		print(command)
		process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
		process.wait()
		if process.returncode != 0 :
			c_error = c_error + 1
			l_error.append((vid_f,' '))
		else:
			c_ok = c_ok + 1 	
			l_ok.append((vid_f, ' '))

		if verbose:
			#this percentage may not be 100 given so videoId doesnot exist
			sucess_progress =  100.0 *(c_ok )/ float(len_total)
			total_percentage =  100.0 *( c_ok + c_error )/ float(len_total)
			print("*************************")
			print("Percent of videos successfully clipped: ", round(sucess_progress, 3))
			print("Percent of total videos processed: ", round(total_percentage, 3))
			print("*************************")
		if (c_ok % 30 == 0):
			write_info(l_ok, c_ok, dest_folder + "/"+"ff_ok_iter_avi.txt")
			write_info(l_error, c_error, dest_folder + "/"+"ff_fail_iter_avi.txt")

		# make sure to ffmpeg run smoothly 
		sleep(3)	
    
	print("Numebe of time that error happened is:", c_error)
	print("Numebe of time that file successfully encoded is:", c_ok)

	return l_ok,c_ok, l_error, c_error
if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	#input json
	parser.add_argument('--video_dir', required=True, help='source video files dir')
	parser.add_argument('--output',     required=True, help='dest folder for clip files')
	parser.add_argument('--verbose', help = 'show progress', default=True )

	args = parser.parse_args()
	avi_files = get_list_files(args.video_dir, f_ext = 'avi')

	# this function returns the list of videos which are sucessfully downloaded
	i_success,c_success, i_fail, c_fail = make_clips(avi_files, args.verbose, args.video_dir, args.output)
	write_info(i_success, c_success, args.output+"/ff_ok.txt")
	write_info(i_fail, c_fail, args.output+"/ff_fail.txt")




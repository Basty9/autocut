import os
import argparse
import subprocess
import click
import datetime
import tempfile
import numpy as np
from scipy.io import wavfile
from cv2 import cv2
from audiotsm import phasevocoder
from audiotsm.io.array import ArrayReader, ArrayWriter
from moviepy.editor import VideoFileClip

parser = argparse.ArgumentParser(description="Speed up silent parts in videos (e.g. lectures).")
parser.add_argument("--input-file", help="the path to a video file (.mp4)", required=True)
parser.add_argument("--output-file", help="the path to the output video (default: will append \"_faster\" to the existing filename)")
parser.add_argument("--silent-speed", default=5, type=int, help="how much silent parts should be sped up (default: 5)")
parser.add_argument("--silent-threshold", default=400, type=int, help="the threshold of what should be considered silent (default: 400)")
parser.add_argument("--show-ffmpeg-output", action="store_true", help="if given, shows ffmpeg output (which is hidden by default)")
args = parser.parse_args()

# Set stdout, stderr for ffmpeg calls and end for any prints before calling ffmpeg
stdout = None if args.show_ffmpeg_output else subprocess.DEVNULL
stderr = None if args.show_ffmpeg_output else subprocess.STDOUT
end = "\n" if args.show_ffmpeg_output else ""

# Utility functions
def get_max_volume(s):
    maxv = np.max(s)
    minv = np.min(s)

    return max(maxv, -minv)

# Create a VideoCapture and get some video properties
capture = cv2.VideoCapture(args.input_file)
video_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
video_length = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
fps = capture.get(cv2.CAP_PROP_FPS)

# Get temporary filenames
original_audio_file = next(tempfile._get_candidate_names()) + ".wav"
video_file = next(tempfile._get_candidate_names()) + ".mp4"
audio_file = next(tempfile._get_candidate_names()) + ".wav"

# Extract audio
print("Extracting audio... ", end=end)
subprocess.call("ffmpeg -i \"{}\" -ab 160k -ac 2 -ar 44100 -vn {}".format(args.input_file, original_audio_file), 
    shell=True, 
    stdout=stdout,
    stderr=stderr)

sample_rate, audio_data = wavfile.read(original_audio_file)
audio_channels = int(audio_data.shape[1])
print("done!")

# Create output writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_output = cv2.VideoWriter(video_file, fourcc, fps, (video_width, video_height))

audio_pointer = 0
modified_audio = np.zeros_like(audio_data, dtype=np.int16)

# Points to the start and end of a silent segment
silent_start = 0
silent_end = -1

# Holds framenumbers instead of whole frame accumulated during silent parts
buffer = []

with click.progressbar(length=video_length, label="Speeding up silent parts of the video...") as bar:
    for _ in bar:
        ret, frame = capture.read()
        # Break if no frames are left
        if ret == False:
            bar.finish()
            break
        
        current_frame = capture.get(cv2.CAP_PROP_POS_FRAMES)
        
        # Get the audio snippet for this frame
        audio_sample_start = round(current_frame / fps * sample_rate)
        audio_sample_end = int(audio_sample_start + sample_rate / fps) # exclusive
        audio_sample = audio_data[audio_sample_start:audio_sample_end]
        
        # This shouldn't happen, but... yeah
        if len(audio_sample) == 0:
            bar.finish()
            break

        if get_max_volume(audio_sample) < args.silent_threshold:
            #buffer.append(frame)
            buffer.append(current_frame)

            silent_end = audio_sample_end
        else:
            if silent_end != -1:
                # Phasevocode silent audio
                silent_sample = audio_data[silent_start:silent_end]

                # audiotsm uses flipped dimensions, so transpose the numpy array
                reader = ArrayReader(silent_sample.transpose())
                writer = ArrayWriter(audio_channels)
                tsm = phasevocoder(reader.channels, speed=args.silent_speed)
                tsm.run(reader, writer)

                # Transpose back to regular dimensions
                phasevocoded_sample = writer.data.transpose()

                # Add silent audio sample
                audio_pointer_end = audio_pointer + len(phasevocoded_sample)
                modified_audio[audio_pointer:audio_pointer_end] = phasevocoded_sample

                # Calculate the position of audio_pointer based on how much
                # sound is needed for the number of video frames added
                audio_pointer += round(len(buffer[::args.silent_speed]) / fps * sample_rate)

                # Write video frames
                #for frame in buffer[::args.silent_speed]:
                #    video_output.write(frame)
                #buffer = []
                for framenumber in buffer[::args.silent_speed]:
                    video_output.write(capture.retrieve(framenumber)[1])
                buffer = []

            # Write video frame
            video_output.write(frame)
            silent_start = audio_sample_end + 1
            silent_end = -1 # indefinite

            # Add audio sample
            audio_pointer_end = audio_pointer + len(audio_sample)
            modified_audio[audio_pointer:audio_pointer_end] = audio_sample
            audio_pointer = audio_pointer_end

# Slice off empty end (as frames have been skipped)
modified_audio = modified_audio[:audio_pointer]
wavfile.write(audio_file, sample_rate, modified_audio)

# Release resources
# For some reason this is necessary, otherwise the resulting mp4 is corrupted
capture.release()
video_output.release()

if args.output_file == None:
    name, ext = os.path.splitext(args.input_file)
    out_file = "{}_faster{}".format(name, ext)
else:
    out_file = args.output_file

# Merge video and audio with ffmpeg
print("Merging video and audio... ", end=end)
if (os.path.exists(out_file)):
    os.remove(out_file)

error = subprocess.call("ffmpeg -i {} -i {} -c:v copy -c:a aac \"{}\"".format(video_file, audio_file, out_file), 
    shell=True, 
    stdout=stdout,
    stderr=stderr)
print("done!")

# Delete temporary files
if error == 0:
    os.remove(original_audio_file)
    os.remove(audio_file)
    os.remove(video_file)

print("The output file is available at {}".format(out_file))

# Compare old and new duration
old_duration = str(datetime.timedelta(seconds=int(video_length/fps)))
new_duration = str(datetime.timedelta(seconds=int(VideoFileClip(out_file).duration)))
print("Old duration: {}, new duration: {}".format(old_duration, new_duration))

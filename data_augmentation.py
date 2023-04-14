import glob, numpy, os, random, soundfile, torch
from scipy import signal
import soundfile as sf
from tqdm import tqdm

# path of the data
data_path = 'dataset/LA/LA/ASVspoof2019_LA_'
protocol_path = 'dataset/LA/LA/ASVspoof2019_LA_cm_protocols/'
musan_path = 'dataset/others/musan/'
rir_path = 'dataset/others/RIRS_NOISES/simulated_rirs/'

# parameter of frames
num_frames = 200

# parameters of the noise
noisetypes = ['noise', 'speech', 'music']
noisesnrs = {'noise': [0, 15], 'speech': [13, 20], 'music': [5, 15]}
noisenum = {'noise': [1, 1], 'speech': [3, 8], 'music': [1, 1]}

# Load and configure augmentation files
def load_noise_rev_files():
    rir_files = glob.glob(os.path.join(rir_path, '*/*/*.wav'))
    noiselist = {}
    augment_files = glob.glob(os.path.join(musan_path, '*/*/*.wav'))
    for file in augment_files:
        # print(file) # dataset/others/musan\music\jamendo\music-jamendo-0151.wav
        # print(file.split('\\')[-3]) # music
        if file.split('\\')[-3] not in noiselist:
            noiselist[file.split('\\')[-3]] = []
        noiselist[file.split('\\')[-3]].append(file)
    return rir_files, noiselist

def data_augmentation(part, rir_files, noiselist):
    if part == 'train':
        protocol = protocol_path + 'ASVspoof2019.LA.cm.' + part + '.trn.txt'
    else:
        protocol = protocol_path + 'ASVspoof2019.LA.cm.' + part + '.trl.txt'
    with open(protocol, 'r') as f:
        audio_info = [info.strip().split() for info in f.readlines()]
    audio_path = data_path + part + '/flac/'

    for audio_details in tqdm(audio_info):
        speaker, filename, _, tag, label = audio_details
        # Read the utterance and randomly select the segment
        audio_handle = audio_path + filename + '.flac'
        audio, sr = soundfile.read(audio_handle)
        length = num_frames * 160 + 240
        if audio.shape[0] <= length:
            shortage = length - audio.shape[0]
            audio = numpy.pad(audio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (audio.shape[0] - length))
        audio = audio[start_frame:start_frame + length]
        audio = numpy.stack([audio], axis=0)

        # Data Augmentation
        augtype = random.randint(0, 5)
        if augtype == 0:  # Original
            audio = audio
        elif augtype == 1:  # Reverberation
            audio = add_rev(rir_files, audio)    
        elif augtype == 2:  # Babble
            audio = add_noise(noiselist, audio, 'speech')
        elif augtype == 3:  # Music
            audio = add_noise(noiselist, audio, 'music')
        elif augtype == 4:  # Noise
            audio = add_noise(noiselist, audio, 'noise')
        elif augtype == 5:  # Television noise
            audio = add_noise(noiselist, audio, 'speech')
            audio = add_noise(noiselist, audio, 'music')
        store_audio(audio_path, audio[0], audio_details, augtype, protocol)

def add_rev(rir_files, audio):
    rir_file = random.choice(rir_files)
    rir, sr = soundfile.read(rir_file)
    rir = numpy.expand_dims(rir.astype(numpy.float), 0)
    rir = rir / numpy.sqrt(numpy.sum(rir ** 2))
    return signal.convolve(audio, rir, mode='full')[:, :num_frames * 160 + 240]


def add_noise(noiselist, audio, noisecat):
    clean_db = 10 * numpy.log10(numpy.mean(audio ** 2) + 1e-4)
    numnoise = noisenum[noisecat]
    noiselist = random.sample(noiselist[noisecat], random.randint(numnoise[0], numnoise[1]))
    noises = []
    for noise in noiselist:
        noiseaudio, sr = soundfile.read(noise)
        length = num_frames * 160 + 240
        if noiseaudio.shape[0] <= length:
            shortage = length - noiseaudio.shape[0]
            noiseaudio = numpy.pad(noiseaudio, (0, shortage), 'wrap')
        start_frame = numpy.int64(random.random() * (noiseaudio.shape[0] - length))
        noiseaudio = noiseaudio[start_frame:start_frame + length]
        noiseaudio = numpy.stack([noiseaudio], axis=0)
        noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2) + 1e-4)
        noisesnr = random.uniform(noisesnrs[noisecat][0], noisesnrs[noisecat][1])
        noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
    noise = numpy.sum(numpy.concatenate(noises, axis=0), axis=0, keepdims=True)
    return noise + audio

def store_audio(audio_path, audio, audio_info, augtype, protocol):
    audio_info[1] = audio_info[1] + '_aug'
    file_name = audio_path + audio_info[1] + '.flac'
    sf.write(file_name, audio, 16000, format='flac', subtype='PCM_16')  # Write out audio as 24bit Flac
    # store the protocol information
    aug_type_info = ['Origin', 'Reverberation','Babble', 'Music', 'Noise', 'Television']
    with open(protocol, 'a') as f:
        f.write(audio_info[0]+' '+audio_info[1]+' '+audio_info[2]+' '+audio_info[3]+' '+audio_info[4]+ ' ' +aug_type_info[augtype] +'\n')

if __name__ == "__main__":
    rir_files, noiselist = load_noise_rev_files()  # Load and configure augmentation files
    parts = ['train', 'dev', 'eval']
    for part in parts:
        data_augmentation(part, rir_files, noiselist)

import kagglehub
print('starting_download')
path = kagglehub.dataset_download('gti-upm/leapgestrecog', force_download=True)
print('download_complete')
print(path)

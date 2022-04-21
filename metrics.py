from pytorch_fid.fid_score import calculate_fid_given_paths

# only works with bs = 1
fid = calculate_fid_given_paths(['datasets/coco/images/val2017/','samples/coco128-30/'], batch_size=1)

print(fid) # 96.28
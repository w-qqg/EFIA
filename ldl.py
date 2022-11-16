
class LDLdataset(Dataset):
    def __init__(self, idx_file, mode, root, vote_num=11):
        print("ds_init_root", root+idx_file)
        self.df = pd.read_table(root + idx_file, sep='\t', header=None)
        self.mode = mode
        self.root = root
        self.vote_num = vote_num

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # fn,vote,div11,std,exist,maxpos
        fn = self.df.iloc[idx][0]
        file_name = self.root + self.df[0][idx]
        img_PIL = Image.open(file_name).convert('RGB')
        if self.mode == "train":
            transformer = transforms.Compose([
                transforms.Resize((256, 256)), transforms.RandomCrop([224, 224]), transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
        else:
            transformer = transforms.Compose([
                transforms.Resize((256, 256)), transforms.CenterCrop([224, 224]), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
        # 存在灰度图，如下处理
        if img_PIL.mode == 'RGB':
            img = transformer(img_PIL)
        else:
            transformer2 = transforms.Compose([
                transforms.Resize((224, 224)), transforms.RandomCrop([224, 224]), transforms.ToTensor(), transforms.Lambda(lambda x: x.repeat(3, 1, 1)), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
            )
            img = transformer2(img_PIL)

#         label = int(self.df.iloc[idx][5])
        label = (torch.tensor(
            eval(self.df.iloc[idx][1])).float())/self.vote_num  # + 1e-6
        return (img, label, fn)

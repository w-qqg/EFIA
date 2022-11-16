def evaluate(loader):
    model.eval()
    top1 = 0
    kl_all = 0
    cheb_all = 0
    itsc_all = 0
    cos_all = 0
    clark_all = 0
    canber_all = 0
    epsilon = 1e-6
    with torch.no_grad():
        for data, ground, fn in tqdm(loader):

            data = data.to(device)

            output = model(data)
            ground = ground.detach().cpu()
            pred = (output).detach().cpu()
            t_data_x = pred

            a = (pred)
            b = (ground)

            kl_all += KL_sum((a + epsilon).log(), b).item()

            cheb_dis = abs(a-b).max(dim=1).values.sum().item()
            cheb_all += cheb_dis

            itsc_dis = torch.min(a, b).sum().item()
            itsc_all += itsc_dis

            clark_all += ((a-b).pow(2)/((a+b).pow(2) + epsilon)
                          ).sum(dim=1).pow(1/2).sum().item()
            canber_all += (abs(a-b)/(a+b + epsilon)).sum().item()
            cos_all += COS((pred), (ground)).sum().item()

            row = 0
            max_val_list = []
            max_val_pos = torch.max(pred, 1)[1]
            gt_values = torch.max(ground, 1)[0]
            for col in max_val_pos:
                max_val_list.append(ground[row, col.item()])
                row += 1
            max_vals = torch.stack(max_val_list, 0)
            correct_nums = (max_vals == gt_values).sum().item()
            top1 += correct_nums

    total = len(loader.dataset)

    Acc = top1/total
    meanKL = kl_all/total
    meanCOS = cos_all/total
    meanCheb = cheb_all/total
    meanItsc = itsc_all/total
    meanClark = clark_all/total
    meanCanber = canber_all/total

    logger.info("an epoch evaluated：\nAcc = %s, Cheb = %s, Clark = %s, Canber = %s, ", str(
        Acc), str(meanCheb), str(meanClark), str(meanCanber))
    logger.info("KLdiv = %s, Cosine = %s, Itsc = %s",
                str(meanKL), str(meanCOS), str(meanItsc))

    # print("Cheb:(<0.25)",meanCheb)
    # print("Clark:(<2.2)",meanClark)
    # print("Canber:(<5.5)",meanCanber)
    # print("KLdiv:(<0.45)",meanKL)
    # print("Cosine:(>0.84)",meanCOS)
    # print("Itsc:(>0.6)",meanItsc)

    return top1/total, meanCOS, meanKL

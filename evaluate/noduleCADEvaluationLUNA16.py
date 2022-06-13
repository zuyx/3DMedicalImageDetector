
import os
import math
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter
import sklearn.metrics as skl_metrics
import numpy as np

from NoduleFinding import NoduleFinding

from tools import csvTools

# matplotlib.rc('xtick', labelsize=18)
# matplotlib.rc('ytick', labelsize=18) 
font = {'family' : 'normal',
        'size'   : 17}

matplotlib.rc('font', **font)
# Evaluation settings
bPerformBootstrapping = True
bNumberOfBootstrapSamples = 1000
bOtherNodulesAsIrrelevant = True
bConfidence = 0.95

seriesuid_label = 'seriesuid'
coordX_label = 'coordX'
coordY_label = 'coordY'
coordZ_label = 'coordZ'
diameter_mm_label = 'diameter_mm'
CADProbability_label = 'probability'

# plot settings
FROC_minX = 0.125 # Mininum value of x-axis of FROC curve
FROC_maxX = 8 # Maximum value of x-axis of FROC curve
bLogPlot = False#True

def generateBootstrapSet(scanToCandidatesDict, FROCImList):
    '''
    Generates bootstrapped version of set
    '''
    imageLen = FROCImList.shape[0]
    
    # get a random list of images using sampling with replacement
    rand_index_im   = np.random.randint(imageLen, size=imageLen)
    FROCImList_rand = FROCImList[rand_index_im]
    
    # get a new list of candidates
    candidatesExists = False
    for im in FROCImList_rand:
        if im not in scanToCandidatesDict:
            continue
        
        if not candidatesExists:
            candidates = np.copy(scanToCandidatesDict[im])
            candidatesExists = True
        else:
            candidates = np.concatenate((candidates,scanToCandidatesDict[im]),axis = 1)

    return candidates

def compute_mean_ci(interp_sens, confidence = 0.95):
    sens_mean = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_lb   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    sens_up   = np.zeros((interp_sens.shape[1]),dtype = 'float32')
    
    Pz = (1.0-confidence)/2.0
    print(interp_sens.shape)
    for i in range(interp_sens.shape[1]):
        # get sorted vector
        vec = interp_sens[:,i]
        vec.sort()

        sens_mean[i] = np.average(vec)
        sens_lb[i] = vec[int(math.floor(Pz*len(vec)))]
        sens_up[i] = vec[int(math.floor((1.0-Pz)*len(vec)))]

    return sens_mean,sens_lb,sens_up

def computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,FROCImList,excludeList,numberOfBootstrapSamples=1000, confidence = 0.95):
    '''
    :param FROCImList: seriesUIDs
    :return:
    '''
    set1 = np.concatenate(([FROCGTList], [FROCProbList], [excludeList]), axis=0)
    
    fps_lists = []
    sens_lists = []
    thresholds_lists = []
    
    FPDivisorList_np = np.asarray(FPDivisorList)
    FROCImList_np = np.asarray(FROCImList) # ？
    
    # Make a dict with all candidates of all scans
    scanToCandidatesDict = {}
    for i in range(len(FPDivisorList_np)):
        seriesuid = FPDivisorList_np[i]
        candidate = set1[:,i:i+1] # 得到的是(GT,Prob,exclude)

        if seriesuid not in scanToCandidatesDict:
            scanToCandidatesDict[seriesuid] = np.copy(candidate)
        else:
            scanToCandidatesDict[seriesuid] = np.concatenate((scanToCandidatesDict[seriesuid],candidate),axis = 1)

    for i in range(numberOfBootstrapSamples):
        # print 'computing FROC: bootstrap %d/%d' % (i,numberOfBootstrapSamples)
        # Generate a bootstrapped set
        btpsamp = generateBootstrapSet(scanToCandidatesDict,FROCImList_np)
        fps, sens, thresholds = computeFROC(btpsamp[0,:],btpsamp[1,:],len(FROCImList_np),btpsamp[2,:])
    
        fps_lists.append(fps)
        sens_lists.append(sens)
        thresholds_lists.append(thresholds)

    # compute statistic
    all_fps = np.linspace(FROC_minX, FROC_maxX, num=10000)
    
    # Then interpolate all FROC curves at this points
    interp_sens = np.zeros((numberOfBootstrapSamples,len(all_fps)), dtype = 'float32')
    for i in range(numberOfBootstrapSamples):
        interp_sens[i,:] = np.interp(all_fps, fps_lists[i], sens_lists[i])
    
    # compute mean and CI
    sens_mean,sens_lb,sens_up = compute_mean_ci(interp_sens, confidence = confidence)

    return all_fps, sens_mean, sens_lb, sens_up

def computeFROC(FROCGTList, FROCProbList, totalNumberOfImages, excludeList):
    # Remove excluded candidates,why remove???
    FROCGTList_local = []
    FROCProbList_local = []
    for i in range(len(excludeList)):
        if excludeList[i] == False:
            FROCGTList_local.append(FROCGTList[i])
            FROCProbList_local.append(FROCProbList[i])
    
    numberOfDetectedLesions = sum(FROCGTList_local) # 正例个数
    totalNumberOfLesions = sum(FROCGTList)
    totalNumberOfCandidates = len(FROCProbList_local) # 样本个数
    fpr, tpr, thresholds = skl_metrics.roc_curve(FROCGTList_local, FROCProbList_local)
    if sum(FROCGTList) == len(FROCGTList): # Handle border case when there are no false positives and ROC analysis give nan values.
        # 标签全是正类
      print ("WARNING, this system has no false positives..")
      fps = np.zeros(len(fpr))
    else:
        # fpr = fp/(fp + tn) 负类预测错误占总负类个数的比例
        # totalNumberOfCandidates - numberOfDetectedLesions = 总样本数 - 正例个数 =  负例个数
        # fps = fp / numberOfImages(for 3d ,a patient all scans is one image)
        fps = fpr * (totalNumberOfCandidates - numberOfDetectedLesions) / totalNumberOfImages
    sens = (tpr * numberOfDetectedLesions) / totalNumberOfLesions
    #print("thresholds shape:",len(thresholds))
    return fps, sens, thresholds

def evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules, CADSystemName, maxNumberOfCADMarks=-1,
                performBootstrapping=False,numberOfBootstrapSamples=1000,confidence = 0.95):
    '''
    function to evaluate a CAD algorithm
    @param seriesUIDs: list of the seriesUIDs of the cases to be processed
    @param results_filename: file with results
    @param outputDir: output directory
    @param allNodules: dictionary with all nodule annotations of all cases, keys of the dictionary are the seriesuids
    @param CADSystemName: name of the CAD system, to be used in filenames and on FROC curve
    '''

    nodOutputfile = open(os.path.join(outputDir,'CADAnalysis_%s.txt'  % CADSystemName),'w')
    nodOutputfile.write("\n")
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("CAD Analysis: %s\n" % CADSystemName)
    nodOutputfile.write((60 * "*") + "\n")
    nodOutputfile.write("\n")

    results = csvTools.readCSV(results_filename)

    allCandsCAD = {}
    
    for seriesuid in seriesUIDs:
        
        # collect candidates from result file
        nodules = {}
        header = results[0]
        
        i = 0
        for result in results[1:]:
            nodule_seriesuid = result[header.index(seriesuid_label)]
            
            # Crop off the heading 0
            '''this part is only for after changing LUNA16 names, remove this may cause error due to different csv version'''
            if nodule_seriesuid[0] == "0":
                nodule_seriesuid  = nodule_seriesuid[1:]
            if nodule_seriesuid[0] == "0":
                nodule_seriesuid  = nodule_seriesuid[1:]
            '''end here'''
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(result, header)
                nodule.candidateID = i
                nodules[nodule.candidateID] = nodule
                i += 1

        if (maxNumberOfCADMarks > 0):
            # number of CAD marks, only keep must suspicous marks

            if len(nodules.keys()) > maxNumberOfCADMarks: # 结节个数大于maxNumberOfCADMarks
                # make a list of all probabilities
                # 按照置信度对PBB排序，只取置信度大于probs[maxNumberOfCADMarks]的PBB
                probs = []
                for keytemp, noduletemp in nodules.items():
                    probs.append(float(noduletemp.CADprobability))
                probs.sort(reverse=True) # sort from large to small
                probThreshold = probs[maxNumberOfCADMarks]
                nodules2 = {}
                nrNodules2 = 0
                for keytemp, noduletemp in nodules.items():
                    if nrNodules2 >= maxNumberOfCADMarks:
                        break
                    if float(noduletemp.CADprobability) > probThreshold:
                        nodules2[keytemp] = noduletemp
                        nrNodules2 += 1

                nodules = nodules2
        
        # print 'adding candidates: ' + seriesuid
        allCandsCAD[seriesuid] = nodules
    # allCansCAD得到的是一个字典，包含所有预测的结节信息
    # open output files
    # 没有被检测到的结节 （没有一个PBB的中心距离于这些结节的中心距离小于范围内）
    nodNoCandFile = open(os.path.join(outputDir, "nodulesWithoutCandidate_%s.txt" % CADSystemName), 'w')
    
    # --- iterate over all cases (seriesUIDs) and determine how
    # often a nodule annotation is not covered by a candidate

    # initialize some variables to be used in the loop
    candTPs = 0
    candFPs = 0
    candFNs = 0
    candTNs = 0
    totalNumberOfCands = 0
    totalNumberOfNodules = 0
    doubleCandidatesIgnored = 0
    irrelevantCandidates = 0
    minProbValue = -1000000000.0 # minimum value of a float
    FROCGTList = []
    FROCProbList = []
    FPDivisorList = []
    excludeList = []
    FROCtoNoduleMap = []
    ignoredCADMarksList = []

    # -- loop over the cases
    for seriesuid in seriesUIDs:
        # get the candidates for this case
        try:
            candidates = allCandsCAD[seriesuid] # 对于一个病人的所有nodules
        except KeyError:
            candidates = {}

        # add to the total number of candidates
        totalNumberOfCands += len(candidates.keys()) # 包含了Excluded Nodules

        # make a copy in which items will be deleted
        candidates2 = candidates.copy()

        # get the nodule annotations on this case
        try:
            noduleAnnots = allNodules[seriesuid] # 真实结节
        except KeyError:
            noduleAnnots = []

        # - loop over the nodule annotations
        for noduleAnnot in noduleAnnots:
            # increment the number of nodules
            if noduleAnnot.state == "Included":
                totalNumberOfNodules += 1 # 真实结节个数

            x = float(noduleAnnot.coordX)
            y = float(noduleAnnot.coordY)
            z = float(noduleAnnot.coordZ)

            # 2. Check if the nodule annotation is covered by a candidate
            # A nodule is marked as detected when the center of mass of the candidate is within a distance R of
            # the center of the nodule. In order to ensure that the CAD mark is displayed within the nodule on the
            # CT scan, we set R to be the radius of the nodule size.
            diameter = float(noduleAnnot.diameter_mm)
            if diameter < 0.0:
              diameter = 10.0
            radiusSquared = pow((diameter / 2.0), 2.0)

            found = False
            noduleMatches = []
            for key, candidate in candidates.items():
                x2 = float(candidate.coordX)
                y2 = float(candidate.coordY)
                z2 = float(candidate.coordZ)
                # 预测PBB与真实PBB的距离
                dist = math.pow(x - x2, 2.) + math.pow(y - y2, 2.) + math.pow(z - z2, 2.)
                if dist < radiusSquared: # PBB的中心与真实结节的中心距离小于半径
                    if (noduleAnnot.state == "Included"):
                        found = True
                        noduleMatches.append(candidate)
                        if key not in candidates2.keys(): # ?
                            print ("!!!!This is strange: CAD mark %s detected two nodules! Check for overlapping nodule annotations, SeriesUID: %s, nodule Annot ID: %s" % (str(candidate.id), seriesuid, str(noduleAnnot.id)))
                        else:
                            del candidates2[key] # 检测到PBB与真实nodule match就删除
                    elif (noduleAnnot.state == "Excluded"): # an excluded nodule
                        # 预测的PBB与Excluded Nodule 匹配，将其删除，并不将其计入FP中
                        if bOtherNodulesAsIrrelevant: #    delete marks on excluded nodules so they don't count as false positives
                            if key in candidates2.keys():
                                irrelevantCandidates += 1
                                # 计入在ignored Nodule文件中
                                ignoredCADMarksList.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate.coordX, candidate.coordY, candidate.coordZ, str(candidate.id), float(candidate.CADprobability)))
                                del candidates2[key]
            # 有多个预测结节与一个真实结节匹配
            if len(noduleMatches) > 1: # double detection
                doubleCandidatesIgnored += (len(noduleMatches) - 1)
            if noduleAnnot.state == "Included": # 只计算Included 结节
                # only include it for FROC analysis if it is included
                # otherwise, the candidate will not be counted as FP, but ignored in the
                # analysis since it has been deleted from the nodules2 vector of candidates
                if found == True: # 当前真实结节有PBB与其匹配，即模型找到了
                    # append the sample with the highest probability for the FROC analysis
                    # 置信度最大的PBB加入FROC 分析中
                    maxProb = None
                    for idx in range(len(noduleMatches)):
                        candidate = noduleMatches[idx]
                        if (maxProb is None) or (float(candidate.CADprobability) > maxProb):
                            maxProb = float(candidate.CADprobability)

                    FROCGTList.append(1.0) # 标签
                    FROCProbList.append(float(maxProb)) # 匹配的结节，取概率最大的
                    FPDivisorList.append(seriesuid)
                    excludeList.append(False)
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%.9f" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(candidate.id), float(candidate.CADprobability)))
                    candTPs += 1
                else:
                    candFNs += 1 # 当前结节没有预测框与其匹配的
                    # append a positive sample with the lowest probability, such that this is added in the FROC analysis
                    FROCGTList.append(1.0)
                    FROCProbList.append(minProbValue) # 不匹配的结节，取概率最低的
                    FPDivisorList.append(seriesuid)
                    excludeList.append(True) #除去没有被检测到的结节吗？？？？
                    FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%.9f,%s,%s" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), int(-1), "NA"))
                    nodNoCandFile.write("%s,%s,%s,%s,%s,%.9f,%s\n" % (seriesuid, noduleAnnot.id, noduleAnnot.coordX, noduleAnnot.coordY, noduleAnnot.coordZ, float(noduleAnnot.diameter_mm), str(-1)))

        # add all false positives to the vectors
        for key, candidate3 in candidates2.items(): #没有被删去的预测结节都是FP
            candFPs += 1
            FROCGTList.append(0.0) # 剩下没有与之匹配的框是负类
            FROCProbList.append(float(candidate3.CADprobability))
            FPDivisorList.append(seriesuid)
            excludeList.append(False) # excluded ？
            FROCtoNoduleMap.append("%s,%s,%s,%s,%s,%s,%.9f" % (seriesuid, -1, candidate3.coordX, candidate3.coordY, candidate3.coordZ, str(candidate3.id), float(candidate3.CADprobability)))

    if not (len(FROCGTList) == len(FROCProbList) and len(FROCGTList) == len(FPDivisorList) and len(FROCGTList) == len(FROCtoNoduleMap) and len(FROCGTList) == len(excludeList)):
        nodOutputfile.write("Length of FROC vectors not the same, this should never happen! Aborting..\n")

    nodOutputfile.write("Candidate detection results:\n")
    nodOutputfile.write("    True positives: %d\n" % candTPs)
    nodOutputfile.write("    False positives: %d\n" % candFPs)
    nodOutputfile.write("    False negatives: %d\n" % candFNs)
    nodOutputfile.write("    True negatives: %d\n" % candTNs)
    nodOutputfile.write("    Total number of candidates（包含Excluded Nodule以及重复检测的Nodules）: %d\n" % totalNumberOfCands)
    nodOutputfile.write("    Total number of nodules: %d\n" % totalNumberOfNodules)

    nodOutputfile.write("    Ignored candidates on excluded nodules: %d\n" % irrelevantCandidates)
    nodOutputfile.write("    Ignored candidates which were double detections on a nodule: %d\n" % doubleCandidatesIgnored)
    if int(totalNumberOfNodules) == 0:
        nodOutputfile.write("    Sensitivity: 0.0\n")
        #nodOutputfile.write("    精度: 0.0\n")
    else:
        nodOutputfile.write("    Sensitivity: %.9f\n" % (float(candTPs) / float(totalNumberOfNodules)))
        #nodOutputfile.write("    精度: %.9f\n" % (float(candTPs) / (float(candTPs) + float(candFPs))))

    nodOutputfile.write("    Average number of candidates per scan（for 3D images  a patient a scan）: %.9f\n" % (float(totalNumberOfCands) / float(len(seriesUIDs))))

    # compute FROC
    fps, sens, thresholds = computeFROC(FROCGTList,FROCProbList,len(seriesUIDs),excludeList)

    #############################################

    if performBootstrapping:
        fps_bs_itp,sens_bs_mean,sens_bs_lb,sens_bs_up = computeFROC_bootstrap(FROCGTList,FROCProbList,FPDivisorList,seriesUIDs,excludeList,
                                                                  numberOfBootstrapSamples=numberOfBootstrapSamples, confidence = confidence)

    nodOutputfile.write("\n##########################\n")
    nodOutputfile.write("    Sens    FROC\n")

    print ("##########################")
    froc_list = [0.125, 0.25, 0.5, 1, 2, 4, 8]
    sum_list = []
    for i in range(len(fps_bs_itp)):
        if round(fps_bs_itp[i],3) in froc_list:
            print(sens_bs_mean[i], round(fps_bs_itp[i],3))
            nodOutputfile.write("    %.5f    %.3f\n" % (sens_bs_mean[i], round(fps_bs_itp[i],3)))
            sum_list.append(sens_bs_mean[i])

    print ("mean sens:",np.mean(np.array(sum_list)))
    nodOutputfile.write("    meansens    {}\n".format(np.mean(np.array(sum_list))))
    
    # Write FROC curve
    with open(os.path.join(outputDir, "froc_%s.txt" % CADSystemName), 'w') as f:
        for i in range(len(sens)):
            f.write("%.9f,%.9f,%.9f\n" % (fps[i], sens[i], thresholds[i]))
    
    # Write FROC vectors to disk as well
    with open(os.path.join(outputDir, "froc_gt_prob_vectors_%s.csv" % CADSystemName), 'w') as f:
        for i in range(len(FROCGTList)):
            f.write("%d,%.9f\n" % (FROCGTList[i], FROCProbList[i]))

    fps_itp = np.linspace(FROC_minX, FROC_maxX, num=10001) # 设置横坐标点
    
    sens_itp = np.interp(fps_itp, fps, sens) # 对每一个点(fp,sen) 按照生成的横坐标 fps_itp，生成纵坐标，如果没有相等的横坐标，就插值
    
    frvvlu = 0
    nxth = 0.125
    for fp, ss in zip(fps_itp, sens_itp):
        if abs(fp - nxth) < 3e-4:
            frvvlu += ss
            nxth *= 2
        if abs(nxth - 16) < 1e-5: break
    print(frvvlu/7, nxth)
#    print(sens_itp[fps_itp==0.125], sens_itp[fps_itp==0.25], sens_itp[fps_itp==0.5], sens_itp[fps_itp==1], sens_itp[fps_itp==2]\
#        ,sens_itp[fps_itp==4],sens_itp[fps_itp==8])
    
    if performBootstrapping:
        # Write mean, lower, and upper bound curves to disk
        with open(os.path.join(outputDir, "froc_%s_bootstrapping.csv" % CADSystemName), 'w') as f:
            f.write("FPrate,Sensivity[Mean],Sensivity[Lower bound],Sensivity[Upper bound]\n")
            for i in range(len(fps_bs_itp)):
                f.write("%.9f,%.9f,%.9f,%.9f\n" % (fps_bs_itp[i], sens_bs_mean[i], sens_bs_lb[i], sens_bs_up[i]))
    else:
        fps_bs_itp = None
        sens_bs_mean = None
        sens_bs_lb = None
        sens_bs_up = None
    # create FROC graphs
    if int(totalNumberOfNodules) > 0:

        graphTitle = str("")
        fig1 = plt.figure()
        ax = plt.gca()
        clr = 'b'
        plt.plot(fps_itp, sens_itp, color=clr, label="%s" % CADSystemName, lw=2)

        if performBootstrapping:
            plt.plot(fps_bs_itp, sens_bs_mean, color=clr, ls='--')
            plt.plot(fps_bs_itp, sens_bs_lb, color=clr, ls=':')  # , label = "lb")
            plt.plot(fps_bs_itp, sens_bs_up, color=clr, ls=':')  # , label = "ub")
            ax.fill_between(fps_bs_itp, sens_bs_lb, sens_bs_up, facecolor=clr, alpha=0.05)

        xmin = FROC_minX
        xmax = FROC_maxX
        #plt.xlim(xmin, xmax)
        plt.ylim(0, 1)
        plt.xlabel('Average number of false positives per scan')
        plt.ylabel('Sensitivity')
        plt.legend(loc='lower right')
        plt.title('FROC performance - %s' % (CADSystemName))

        if bLogPlot:
            plt.xscale('log', base=2)
            ax.xaxis.set_major_formatter(FixedFormatter([0.125, 0.25, 0.5, 1, 2, 4, 8]))

        # set your ticks manually
        ax.xaxis.set_ticks(np.linspace(1, 7 , 7))
        ax.set_xticklabels(('0.125', '0.25', '0.5', '1', '2', '4', '8'))
        #ax.xaxis.set_ticks()
        #        ax.yaxis.set_ticks(np.arange(0.5, 1, 0.1))

        ax.yaxis.set_ticks(np.arange(0, 1.1, 0.1))
        plt.grid(b=True, which='both')
        plt.tight_layout()

        plt.savefig(os.path.join(outputDir, "froc_%s.png" % CADSystemName), bbox_inches=0, dpi=300)

    return (fps, sens, thresholds, fps_bs_itp, sens_bs_mean, sens_bs_lb, sens_bs_up)

def getNodule(annotation, header, state = ""):
    nodule = NoduleFinding()
    nodule.coordX = annotation[header.index(coordX_label)]
    nodule.coordY = annotation[header.index(coordY_label)]
    nodule.coordZ = annotation[header.index(coordZ_label)]
    
    if diameter_mm_label in header:
        nodule.diameter_mm = annotation[header.index(diameter_mm_label)]
    
    if CADProbability_label in header:
        nodule.CADprobability = annotation[header.index(CADProbability_label)]
    
    if not state == "":
        nodule.state = state

    return nodule
    
def collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs):
    allNodules = {}
    noduleCount = 0
    noduleCountTotal = 0
    
    for seriesuid in seriesUIDs:
        # print 'adding nodule annotations: ' + seriesuid
        
        nodules = []
        numberOfIncludedNodules = 0
        
        # add included findings
        header = annotations[0]
        for annotation in annotations[1:]:
            nodule_seriesuid = annotation[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                # 在annotation csv中找到seriesuid对应的所有的nodule
                nodule = getNodule(annotation, header, state = "Included")
                nodules.append(nodule)
                numberOfIncludedNodules += 1
        
        # add excluded findings
        header = annotations_excluded[0]
        for annotation in annotations_excluded[1:]:            
            nodule_seriesuid = annotation[header.index(seriesuid_label)]
            
            if seriesuid == nodule_seriesuid:
                nodule = getNodule(annotation, header, state = "Excluded")
                nodules.append(nodule)
            
        allNodules[seriesuid] = nodules # 找到一个病人的所有结节（包含includeed和excluded）
        noduleCount      += numberOfIncludedNodules
        noduleCountTotal += len(nodules) # 所有
    
    print ('Total number of included nodule annotations: ' + str(noduleCount))
    print ('Total number of nodule annotations: ' + str(noduleCountTotal))
    return allNodules
    
    
def collect(annotations_filename,annotations_excluded_filename,seriesuids_filename):
    annotations          = csvTools.readCSV(annotations_filename)
    annotations_excluded = csvTools.readCSV(annotations_excluded_filename)
    seriesUIDs_csv = csvTools.readCSV(seriesuids_filename)
    
    seriesUIDs = [] # subseti中所有病人的id

    for seriesUID in seriesUIDs_csv:
        seriesUIDs.append(seriesUID[0])

    allNodules = collectNoduleAnnotations(annotations, annotations_excluded, seriesUIDs)
    
    return (allNodules, seriesUIDs) # 返回这个subset中所有病人的结节字典、病人序列号
    
    
def noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputDir):
    '''
    function to load annotations and evaluate a CAD algorithm
    @param annotations_filename: list of annotations
    @param annotations_excluded_filename: list of annotations that are excluded from analysis
    @param seriesuids_filename: list of CT images in seriesuids
    @param results_filename: list of CAD marks with probabilities
    @param outputDir: output directory
    '''
    
    print ("annotations_filename:",annotations_filename)
    
    (allNodules, seriesUIDs) = collect(annotations_filename, annotations_excluded_filename, seriesuids_filename)

    # 找到所有病人的结节信息：allNodules
    # 模型预测的所有结节信息：results_filename
    return evaluateCAD(seriesUIDs, results_filename, outputDir, allNodules,
                os.path.splitext(os.path.basename(results_filename))[0],
                maxNumberOfCADMarks=100, performBootstrapping=bPerformBootstrapping,
                numberOfBootstrapSamples=bNumberOfBootstrapSamples, confidence=bConfidence)

'''
if __name__ == '__main__':
    ################################Need to change this part#######################################################
    #fold = 0
    #data_dir = '/hdd/hdd1/zyx/datasets/luna16/'
    #annotations_dir = data_dir + 'annotations/10FoldCsvFiles/fold' + str(fold) + '/'
    #annotations_filename = annotations_dir + 'annotations.csv'  # path for ground truth annotations for the fold
    #annotations_excluded_filename = annotations_dir + 'annotations_excluded.csv'  # path for excluded annotations for the fold
    #seriesuids_filename = annotations_dir + 'seriesuids.csv'  # path for seriesuid for the fold
    #results_filename = annotations_filename
    #model = 'res18_old'
    #nmsthresh = 0.1
    #outputdir = outputdir = './bboxOutput/' + model + '/nms' + str(nmsthresh) + '/'
    data_dir = '/home/ren/zyx/datasets/luna16/'

    annotations_filename = data_dir + 'labels/annos.csv'
    annotations_excluded_filename = data_dir + 'labels/excluded.csv'
    seriesuids_filename = data_dir + 'labels/test_luna_ID_fold6.csv'

    outputDir = './TEST/detep'
    results_filename = './predanno/predanno0.29.csv'
    outputdir = outputDir
    froc_outdir = outputDir
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    noduleCADEvaluation(annotations_filename,annotations_excluded_filename,seriesuids_filename,results_filename,outputdir)
    print ("Finished!")
'''
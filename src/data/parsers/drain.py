"""
Drain log parsing algorithm.
Original: LogPAI team (MIT License) — adapted from CSCLog/utils/Drain.py.
"""
import hashlib
import os
from datetime import datetime

import numpy as np
import pandas as pd
import regex as re


class Logcluster:
    def __init__(self, logTemplate="", logIDL=None):
        self.logTemplate = logTemplate
        self.logIDL = logIDL if logIDL is not None else []


class Node:
    def __init__(self, childD=None, depth=0, digitOrtoken=None):
        self.childD = childD if childD is not None else {}
        self.depth = depth
        self.digitOrtoken = digitOrtoken


class LogParser:
    def __init__(
        self,
        log_format: str,
        indir: str = "./",
        outdir: str = "./result/",
        depth: int = 4,
        st: float = 0.4,
        maxChild: int = 100,
        rex: list = None,
        keep_para: bool = True,
    ):
        self.path = indir
        self.depth = depth - 2
        self.st = st
        self.maxChild = maxChild
        self.logName = None
        self.savePath = outdir
        self.df_log = None
        self.log_format = log_format
        self.rex = rex or []
        self.keep_para = keep_para

    def hasNumbers(self, s):
        return any(c.isdigit() for c in s)

    def treeSearch(self, rn, seq):
        seqLen = len(seq)
        if seqLen not in rn.childD:
            return None
        parentn = rn.childD[seqLen]
        currentDepth = 1
        for token in seq:
            if currentDepth >= self.depth or currentDepth > seqLen:
                break
            if token in parentn.childD:
                parentn = parentn.childD[token]
            elif "<*>" in parentn.childD:
                parentn = parentn.childD["<*>"]
            else:
                return None
            currentDepth += 1
        return self.fastMatch(parentn.childD, seq)

    def addSeqToPrefixTree(self, rn, logClust):
        seqLen = len(logClust.logTemplate)
        if seqLen not in rn.childD:
            rn.childD[seqLen] = Node(depth=1, digitOrtoken=seqLen)
        parentn = rn.childD[seqLen]
        currentDepth = 1
        for token in logClust.logTemplate:
            if currentDepth >= self.depth or currentDepth > seqLen:
                if len(parentn.childD) == 0:
                    parentn.childD = [logClust]
                else:
                    parentn.childD.append(logClust)
                break
            if token not in parentn.childD:
                if not self.hasNumbers(token):
                    if "<*>" in parentn.childD:
                        if len(parentn.childD) < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD["<*>"]
                    else:
                        if len(parentn.childD) + 1 < self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken=token)
                            parentn.childD[token] = newNode
                            parentn = newNode
                        elif len(parentn.childD) + 1 == self.maxChild:
                            newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                            parentn.childD["<*>"] = newNode
                            parentn = newNode
                        else:
                            parentn = parentn.childD["<*>"]
                else:
                    if "<*>" not in parentn.childD:
                        newNode = Node(depth=currentDepth + 1, digitOrtoken="<*>")
                        parentn.childD["<*>"] = newNode
                        parentn = newNode
                    else:
                        parentn = parentn.childD["<*>"]
            else:
                parentn = parentn.childD[token]
            currentDepth += 1

    def seqDist(self, seq1, seq2):
        assert len(seq1) == len(seq2)
        simTokens = sum(1 for t1, t2 in zip(seq1, seq2) if t1 != "<*>" and t1 == t2)
        numOfPar = sum(1 for t in seq1 if t == "<*>")
        return float(simTokens) / len(seq1), numOfPar

    def fastMatch(self, logClustL, seq):
        maxSim, maxNumOfPara, maxClust = -1, -1, None
        for lc in logClustL:
            curSim, curNum = self.seqDist(lc.logTemplate, seq)
            if curSim > maxSim or (curSim == maxSim and curNum > maxNumOfPara):
                maxSim, maxNumOfPara, maxClust = curSim, curNum, lc
        return maxClust if maxSim >= self.st else None

    def getTemplate(self, seq1, seq2):
        return [w if w == seq2[i] else "<*>" for i, w in enumerate(seq1)]

    def outputResult(self, logClustL):
        log_templates = [""] * self.df_log.shape[0]
        log_templateids = [""] * self.df_log.shape[0]
        df_events = []
        for lc in logClustL:
            template_str = " ".join(lc.logTemplate)
            template_id = hashlib.md5(template_str.encode("utf-8")).hexdigest()[:8]
            for logID in lc.logIDL:
                log_templates[logID - 1] = template_str
                log_templateids[logID - 1] = template_id
            df_events.append([template_id, template_str, len(lc.logIDL)])

        self.df_log["EventId"] = log_templateids
        self.df_log["EventTemplate"] = log_templates
        if self.keep_para:
            self.df_log["ParameterList"] = self.df_log.apply(self._get_parameter_list, axis=1)

        os.makedirs(self.savePath, exist_ok=True)
        self.df_log.to_csv(
            os.path.join(self.savePath, self.logName + "_structured.csv"), index=False
        )

        occ_dict = dict(self.df_log["EventTemplate"].value_counts())
        df_event = pd.DataFrame()
        df_event["EventTemplate"] = self.df_log["EventTemplate"].unique()
        df_event["EventId"] = df_event["EventTemplate"].map(
            lambda x: hashlib.md5(x.encode("utf-8")).hexdigest()[:8]
        )
        df_event["Occurrences"] = df_event["EventTemplate"].map(occ_dict)
        df_event.to_csv(
            os.path.join(self.savePath, self.logName + "_templates.csv"),
            index=False,
            columns=["EventId", "EventTemplate", "Occurrences"],
        )

    def parse(self, logName: str):
        print(f"Parsing file: {os.path.join(self.path, logName)}")
        start_time = datetime.now()
        self.logName = logName
        rootNode = Node()
        logCluL = []
        self.load_data()
        for idx, line in self.df_log.iterrows():
            logID = line["LineId"]
            logmessageL = self.preprocess(line["Content"]).strip().split()
            matchCluster = self.treeSearch(rootNode, logmessageL)
            if matchCluster is None:
                newCluster = Logcluster(logTemplate=logmessageL, logIDL=[logID])
                logCluL.append(newCluster)
                self.addSeqToPrefixTree(rootNode, newCluster)
            else:
                newTemplate = self.getTemplate(logmessageL, matchCluster.logTemplate)
                matchCluster.logIDL.append(logID)
                if " ".join(newTemplate) != " ".join(matchCluster.logTemplate):
                    matchCluster.logTemplate = newTemplate
            if (idx + 1) % 10000 == 0:
                print(f"  Processed {(idx+1)*100.0/len(self.df_log):.1f}%")
        self.outputResult(logCluL)
        print(f"Parsing done. [{datetime.now() - start_time}]")
        return self.df_log

    def load_data(self):
        headers, regex = self._generate_logformat_regex(self.log_format)
        self.df_log = self._log_to_dataframe(
            os.path.join(self.path, self.logName), regex, headers
        )

    def preprocess(self, line: str) -> str:
        for pattern in self.rex:
            line = re.sub(pattern, "<*>", line)
        return line

    def _log_to_dataframe(self, log_file, regex, headers):
        log_messages, linecount = [], 0
        with open(log_file, "r", encoding="utf-8", errors="replace") as fin:
            for line in fin:
                try:
                    match = regex.search(line.strip())
                    if match:
                        log_messages.append([match.group(h) for h in headers])
                        linecount += 1
                except Exception:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        if "LineId" not in logdf.columns:
            logdf.insert(0, "LineId", range(1, linecount + 1))
        return logdf

    def _generate_logformat_regex(self, logformat):
        headers = []
        splitters = re.split(r"(<[^<>]+>)", logformat)
        pattern = ""
        for k, part in enumerate(splitters):
            if k % 2 == 0:
                pattern += re.sub(r" +", r"\\s+", part)
            else:
                header = part.strip("<>")
                pattern += f"(?P<{header}>.*?)"
                headers.append(header)
        return headers, re.compile("^" + pattern + "$")

    def _get_parameter_list(self, row):
        template_regex = re.sub(r"<.{1,5}>", "<*>", row["EventTemplate"])
        if "<*>" not in template_regex:
            return []
        template_regex = re.sub(r"([^A-Za-z0-9])", r"\\\1", template_regex)
        template_regex = re.sub(r"\\ +", r"\\s+", template_regex)
        template_regex = "^" + template_regex.replace("\\<\\*\\>", "(.*?)") + "$"
        parameter_list = re.findall(template_regex, row["Content"])
        parameter_list = parameter_list[0] if parameter_list else ()
        return list(parameter_list) if isinstance(parameter_list, tuple) else [parameter_list]

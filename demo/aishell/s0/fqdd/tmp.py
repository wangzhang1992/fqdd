#!/usr/bin/env python3

"""
@author: zw
语音识别API的HTTP服务器程序
"""

import os
import http.server
import urllib
import socket
import subprocess
import json
import time
from audio_transf import trans_audio_types
from inference import inference

HOST = '0.0.0.0'
PORT = 8080
ADDR = (HOST, PORT)

asr_param = json.load(open('conf/asr_param.json', 'r', encoding='utf-8'))


class ASR_process:
    def __init__(self, param_json, sendstr):
        self.param_json = param_json
        self.send = sendstr

    # store wav data
    def audio_get(self):

        datas = self.param_json['audio_data']
        audio_type = self.param_json['audio_type']
        model_type = self.param_json['model_type']
        data_type = self.param_json['am_model']
        fs = 16000 if 'Service' not in model_type else 8000

        utt = str(time.time_ns())
        save_path = 'wav_data/wav/'
        tmp_path = 'wav_data/tmp/'
        os.makedirs(tmp_path, exist_ok=True)
        os.makedirs(save_path, exist_ok=True)

        tmp_wav = tmp_path + '/' + utt + '.' + audio_type
        save_wav = save_path + '/' + utt + '.wav'
        try:
            open(tmp_wav, 'wb').write(datas)
            # trans_audio_types(audio_file, out_file, fs=16000, input_type='wav', output_type='wav')
            # 由于目前客服模型输入为16k 因此此处fs 不做变化,使用16k。
            trans_audio_types(tmp_wav, save_wav, input_type=audio_type)
        except:
            self.send.write('audio error')
        finally:
            if os.path.exists(tmp_wav):
                os.remove(tmp_wav)
        return utt, save_wav

    # prepare scp
    def data_scp(self, utts, save_wav):
        if not os.path.exists('data/' + utts):
            os.makedirs('data/' + utts)
        with open('data/' + utts + '/wav.scp', 'w', encoding='utf-8') as wf:
            wf.write(utts + ' ' + save_wav + '\n')

        with open('data/' + utts + '/utt2spk', 'w', encoding='utf-8') as wf:
            wf.write(utts + ' ' + utts + '\n')

        with open('data/' + utts + '/spk2utt', 'w', encoding='utf-8') as wf:
            wf.write(utts + ' ' + utts + '\n')

    def decode(self, utts, model_param):
        asr_res = ''
        try:
            am_path = asr_param['Language'][model_param['language']][model_param['model_type']][model_param['am_model']]
            lm_path = asr_param['Lm'][model_param['language']][model_param['model_type']][model_param['lm']]
            conf_path = asr_param['Conf'][model_param['language']][model_param['model_type']][model_param['am_model']]
            language = model_param['language']
            punt_model = asr_param['PostProcessor']['Punt'][language]
        except:
            self.send.write(str("param error"))
        try:
            a = subprocess.run("sh scripts/asr_decode.sh " + utts + ' ' + am_path + ' ' + lm_path + ' ' + conf_path,
                               shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8")
            print(a)
            if a.returncode == 0:
                pass
            else:
                if os.path.exists('result/' + utts + '/rescore/asr.1.log'):
                    os.system('cat result/' + utts + '/rescore/asr.*.log > result/' + utts + '/asr.log.txt')
                elif os.path.exists('result/' + utts + '/decode/asr.1.log'):
                    os.system('cat result/' + utts + '/decode/asr.*.log > result/' + utts + '/asr.log.txt')
                else:
                    return
        except:
            self.send(str("asr decode error, please check audio"))
        if os.path.exists('result/' + utts + '/asr.log.txt'):
            asr_res = asr_text('result/' + utts + '/asr.log.txt')

        return asr_res, language

    def asr_text(self, asrtxt):
        result = ''
        with open(asrtxt, 'r', encoding='utf8') as rf:
            for line in rf:
                # if is_contain_chinese(line.strip()) and "nnet-batch-compute" not in line.strip() and "nnet3-latgen-faster" not in line.strip():
                if "LOG" not in line.strip() and "nnet3-latgen-faster" not in line.strip() and "lattice-scale" not in line.strip() and "#" not in line.strip() and 'apply-cmvn' not in line.strip():
                    lines = line.strip().split(' ')
                    result = ' '.join(lines[1:])
        return result

    def punt_text(self, text, language):

        try:
            punt_res = inference(text, language)
        except:
            self.send.write("punt error")
        finally:
            return punt_re

    def process_asr(self):
        utt, save_wav = self.audio_get()
        self.data_scp(utt, save_wav)
        asr_res, language = self.decode(utt, self.param_json )
        text = self.asr_text(asr_res)
        punt_res = self.punt_text(text, language)
        return punt_res


class ASRTHTTPHandle(http.server.BaseHTTPRequestHandler):
    def setup(self):
        self.request.settimeout(10)  # 设定超时时间10秒
        http.server.BaseHTTPRequestHandler.setup(self)

    def _set_response(self):
        self.send_response(200)  # 设定HTTP服务器请求状态200
        self.send_header('Content-type', 'text/html')  # 设定HTTP服务器请求内容格式
        self.end_headers()

    def check_permit(self, usr_ID, usr_PW):

        if usr_ID == "admin" and usr_PW == "123456":
            response_str = "congratulation, you have connect with server\r\nnow, you can do something else"
            websocket.send(str(response_str))
            return True
        else:
            response_str = "sorry, your username or password is wrong, please submit again"
            self.wfile.write(response_str)

            return False

    def recv_msg(self):

        param_text = self.rfile.read(int(self.headers['content-length']))
        param_json = eval(param_text)
        return param_json

    def do_GET(self):
        buf = 'ASR_SpeechRecognition API'
        self.protocal_version = 'HTTP/1.1'

        self._set_response()

        buf = bytes(buf, encoding="utf-8")  # 编码转换
        self.wfile.write(buf)  # 写出返回内容

    def do_POST(self):
        '''
        处理通过POST方式传递过来并接收的输入数据并计算和返回结果
        '''
        path = self.path
        # 获取post提交的数据
        param_json = recv_msg()
        if not check_permit(param_json['usr_ID'], param_json['usr_PW']):
            return

        ASR_process(param_json, self.wfile).process_asr()

        self._set_response()

        print(result)
        result = bytes(result, encoding="utf-8")
        self.wfile.write(result)


class HTTPServerV6(http.server.HTTPServer):
    address_family = socket.AF_INET6


def start_server(ip, port):
    if (':' in ip):
        http_server = HTTPServerV6((ip, port), ASRTHTTPHandle)
    else:
        http_server = http.server.HTTPServer((ip, int(port)), ASRTHTTPHandle)

    print('服务器已开启')

    try:
        http_server.serve_forever()  # 设置一直监听并接收请求
    except KeyboardInterrupt:
        pass
    http_server.server_close()
    print('HTTP server closed')


if __name__ == '__main__':
    start_server('', 8080)  # For IPv4 Network Only
    # start_server('::', 20000) # For IPv6 Network
package main

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"time"
)

const (
	douBaoVoiceCloneHost = "https://openspeech.bytedance.com"
	douBaoVoice          = "https://open.volcengineapi.com"
	Region               = "cn-north-1"
	Service              = "speech_saas_prod"
	Version              = "2023-11-07"
)

type DouBaoResp struct {
	Result struct {
		Statuses *[]struct {
			SpeakerID string `json:"SpeakerID"`
		} `json:"Statuses"`
	} `json:"Result"`
}

func parseSpeakerID(responseRaw []byte) (string, error) {

	var resp DouBaoResp
	if err := json.Unmarshal(responseRaw, &resp); err != nil {
		fmt.Print("ddddddddddddddddddddd")
		return "", fmt.Errorf("json unmarshal error: %w", err)
	}

	if resp.Result.Statuses != nil && len(*resp.Result.Statuses) > 0 {
		return (*resp.Result.Statuses)[0].SpeakerID, nil
	}

	return "", fmt.Errorf("no SpeakerID found")
}

// 火山引擎音色复刻 sign
// https://www.volcengine.com/docs/6561/1305191

func hmacSHA256(key []byte, content string) []byte {
	mac := hmac.New(sha256.New, key)
	mac.Write([]byte(content))
	return mac.Sum(nil)
}

func getSignedKey(secretKey, date, region, service string) []byte {
	kDate := hmacSHA256([]byte(secretKey), date)
	kRegion := hmacSHA256(kDate, region)
	kService := hmacSHA256(kRegion, service)
	kSigning := hmacSHA256(kService, "request")

	return kSigning
}

func hashSHA256(data []byte) []byte {
	hash := sha256.New()
	if _, err := hash.Write(data); err != nil {
		fmt.Print("[hashSHA256] input hash err:", err.Error())
	}

	return hash.Sum(nil)
}

// 火山引擎音色复刻 获取一个未使用的音色 speakerid
func getUnusedDouBaoSpeakerID() (string, error) {
	// 1. 构建请求
	//queries.Set("Action", Action)
	Addr := "https://open.volcengineapi.com"
	queries := make(url.Values)
	queries.Set("Version", "2023-11-07")
	queries.Set("Action", "ListMegaTTSTrainStatus")

	requestAddr := fmt.Sprintf("%s%s?%s", Addr, "/", queries.Encode())
	fmt.Print("[getUnusedDouBaoSpeakerID] request addr：", requestAddr)

	// var body []byte = (`{"AppID": "6949418957","State":"Unknown"}`)
	body := []byte(`{"AppID": "2563894307","State":"Unknown"}`)
	req, err := http.NewRequest("POST", requestAddr, bytes.NewBuffer(body))
	if err != nil {
		fmt.Print("[getUnusedDouBaoSpeakerID] bad request: ", err)
		return "", fmt.Errorf("bad request: %w", err)
	}
	// 2. 构建签名材料
	now := time.Now()
	date := now.UTC().Format("20060102T150405Z")
	authDate := date[:8]
	req.Header.Set("X-Date", date)
	payload := hex.EncodeToString(hashSHA256(body))
	req.Header.Set("X-Content-Sha256", payload)
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")

	queryString := strings.Replace(queries.Encode(), "+", "%20", -1)
	signedHeaders := []string{"host", "x-date", "x-content-sha256", "content-type"}
	var headerList []string
	for _, header := range signedHeaders {
		if header == "host" {
			headerList = append(headerList, header+":"+req.Host)
		} else {
			v := req.Header.Get(header)
			headerList = append(headerList, header+":"+strings.TrimSpace(v))
		}
	}
	headerString := strings.Join(headerList, "\n")

	canonicalString := strings.Join([]string{
		"POST",
		"/",
		queryString,
		headerString + "\n",
		strings.Join(signedHeaders, ";"),
		payload,
	}, "\n")
	fmt.Print("[getUnusedDouBaoSpeakerID] canonical string:", canonicalString)

	hashedCanonicalString := hex.EncodeToString(hashSHA256([]byte(canonicalString)))
	fmt.Print("[getUnusedDouBaoSpeakerID] hashed canonical string::", hashedCanonicalString)

	credentialScope := authDate + "/" + "cn-north-1" + "/" + "speech_saas_prod" + "/request"
	signString := strings.Join([]string{
		"HMAC-SHA256",
		date,
		credentialScope,
		hashedCanonicalString,
	}, "\n")

	fmt.Print("[getUnusedDouBaoSpeakerID] sign string:", signString)

	// 3. 构建认证请求头
	signedKey := getSignedKey(SecretAccessKey, authDate, Region, Service)
	signature := hex.EncodeToString(hmacSHA256(signedKey, signString))
	fmt.Print("[getUnusedDouBaoSpeakerID] signature:", signature)

	authorization := "HMAC-SHA256" +
		" Credential=" + AccessKeyID + "/" + credentialScope +
		", SignedHeaders=" + strings.Join(signedHeaders, ";") +
		", Signature=" + signature
	req.Header.Set("Authorization", authorization)
	fmt.Print("[getUnusedDouBaoSpeakerID] authorization:", authorization)

	clientHttp := &http.Client{}
	res, err := clientHttp.Do(req)
	if err != nil {
		fmt.Print("[getUnusedDouBaoSpeakerID]dump request err: ", err)
		return "", fmt.Errorf("dump request err: %w", err)
	}
	defer res.Body.Close()

	// 读取响应内容
	respbody, _ := ioutil.ReadAll(res.Body)
	if err != nil {
		fmt.Print("[RegisterVoice] Read response error:", err)
		return "", err
	}
	speaker, _ := parseSpeakerID(respbody)
	fmt.Print("a ...any", speaker)

	// if err != nil {
	// 	fmt.Print("[getUnusedDouBaoSpeakerID] dump response err: ", err)
	// 	return "", fmt.Errorf("dump response err: %w", err)
	// }

	return "", err
}

func main() {
	_, _ = getUnusedDouBaoSpeakerID()

	// query1 := make(url.Values)
	// query1.Set("Action", "ListMegaTTSTrainStatus")
	// //ListMegaTTSTrainStatus 不指定音色
	// doRequest(http.MethodPost, query1, []byte(`{"AppID": "2563894307","State":"Unknown"}`))
	//ListMegaTTSTrainStatus 指定音色
	// doRequest(http.MethodPost, query1, []byte(`{"AppID": "填入真实值", "SpeakerIDs": ["填入真实值"]}`))

	// query2 := make(url.Values)
	// query2.Set("Action", "ActivateMegaTTSTrainStatus")
	// doRequest(http.MethodPost, query2, []byte(`{"AppID": "填入真实值", "SpeakerIDs": ["填入真实值"]}`))
}

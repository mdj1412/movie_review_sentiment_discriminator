import argparse
import urllib.request
from bs4 import BeautifulSoup
import pandas as pd
import datetime
from selenium import webdriver
import re
from tqdm import tqdm


import netizen_score
import review_scrap
import review_scrap_version





# 영화 제목 가져오기
def get_title(movie_id):
    url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code={}'.format(movie_id)

    # html 가져오기
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    text = soup.select('div[class=mv_info_area] div[class=mv_info] h3[class=h_movie] a')[0].text
    return text



# 영화 개봉 날짜 가져오기
def get_release_date(movie_id):
    url = 'https://movie.naver.com/movie/bi/mi/basic.naver?code={}'.format(movie_id)

    # html 가져오기
    html = urllib.request.urlopen(url).read()
    soup = BeautifulSoup(html, 'html.parser')

    release_date=0
    try:
        # row = soup.select('dl[class=info_spec] dd p span')[-1].select('a')
        row = soup.select('dl[class=info_spec] dd p')[0].select('span')[-1].select('a')
        release_date = row[0].text.strip() + row[1].text.strip()
    except Exception as e:
        print("get_release_date() ")

        from IPython import embed; embed()

        # 여기서 멈추면 밑에꺼 실행
        # row = soup.select('h3[class=h_movie] strong[class=h_movie2]')[0].text.strip()
        # release_date = row[-4:] + '0000'
        # return e # 이건 제외

    return release_date




def main():
    # 최근 review data 가져오기  10,000
    # review_data = netizen_score.get_review_data(start_page=1, end_page=10) # 최대 1000페이지
    # df = pd.DataFrame(review_data)
    # df['release_date'] = '9999.99.99'   # 최근 출시된 영화는 아니지만 최근에 업로드된 리뷰

    df = pd.DataFrame()


    # 최근 인기 영화 10개 movie id 가져오기
    movie_id_set = load_movie_id(start=20050401, end=20230101)

    movie_id_set.add(167651)#극한직업
    movie_id_set.add(93756)#명량
    movie_id_set.add(161967)#기생충
    movie_id_set.add(85579)#신과함께-죄와 벌
    movie_id_set.add(102875)#국제시장
    movie_id_set.add(121048)#암살
    movie_id_set.add(146469)#택시운전사
    movie_id_set.add(130966)#부산행
    movie_id_set.add(101901)#변호인
    movie_id_set.add(136900)#어벤져스: 엔드게임
    movie_id_set.add(136873)#겨울왕국 2
    movie_id_set.add(39841)#괴물

    print("# of movie_id_set : ", len(movie_id_set))


    # 10,000 X 10
    i = 0
    for movie_id in tqdm(movie_id_set):
        try:
            review_dict_high = review_scrap_version.get_review_data(movie_id, version='highest', start_page=1, end_page=15, spoiler='N')
            review_dict_low = review_scrap_version.get_review_data(movie_id, version='lowest', start_page=1, end_page=15, spoiler='N')
            # review_dict = review_scrap.get_review_data(movie_id, start_page=1, end_page=500, spoiler='N')
            
            if review_dict_high == 'Failed: please check end_page' or review_dict_low == 'Failed: please check end_page':
                continue

            if type(review_dict_high) != list or type(review_dict_low) != list:
                from IPython import embed; embed()

            # dictionary -> dataframe
            movie_id_df_high = pd.DataFrame(review_dict_high)
            movie_id_df_low = pd.DataFrame(review_dict_low)
            movie_id_df = pd.concat([movie_id_df_high, movie_id_df_low], ignore_index=True)

            # 해당 movie_id title 뽑아오기
            movie_id_df['title'] = get_title(movie_id)
            # 해당 movie_id release_date 뽑아오기 (개봉날짜)
            movie_id_df['release_date'] = get_release_date(movie_id)

            # dataframe 합치기
            df = pd.concat([df, movie_id_df], ignore_index=True)

        except Exception as e:
            print("Error : main() ")
            from IPython import embed; embed()
            # return e
    

    # Current Time
    new = datetime.datetime.now()
    time = new.strftime("%Y_%m_%d_%H-%M-%S")
    print("time:", time)

    df.to_csv('./Dataset/save_{}.tsv'.format(time), index=True, sep='\t')
    print(df)







def load_movie_id(start=20050401, end=20230101):
    movie_id_set = set()

    date = start
    while date <= end:
        try:
            url = "https://movie.naver.com/movie/sdb/rank/rmovie.naver?sel=cnt&tg=0&date={}".format(date)

            # html 가져오기
            html = urllib.request.urlopen(url).read()
            soup = BeautifulSoup(html, 'html.parser')

            for i in range(1, 11):
                raw = soup.select('table[class=list_ranking] tbody')[0].select('tr')
                movie_id = raw[i].select('a')[0].attrs['href'].split('code=')[1]
                movie_id_set.add(movie_id)
            
            date += 300
            if str(date)[4:6] == '13':
                year = int(date / 10000 + 1)
                date = int(year * 10000 + 101)

        except Exception as e:
            print("Error : load_movie_id() ")
            cont=0 # 계속 진행할 것인지
            from IPython import embed; embed()
            if cont==1: continue
            else: return e

    return movie_id_set











def current_top3_movie():
    movie_id_list = []
    driver = webdriver.Chrome(executable_path='/Users/mindongjun/Desktop/Kaggle/창원_공모전/chromedriver')

    # 1 ~ 10
    for num in range(1, 4):
        try:
            # 주소 이동 ( 네이버 박스오피스 )
            driver.get("https://search.naver.com/search.naver?where=nexearch&sm=top_sug.pre&fbm=1&acr=5&acq=%EB%B0%95%EC%8A%A4%EC%98%A4%ED%94%BC%EC%8A%A4&qdt=0&ie=utf8&query=%EB%B0%95%EC%8A%A4%EC%98%A4%ED%94%BC%EC%8A%A4")

            # Go to movie homepage
            xpath = "/html/body/div[3]/div[2]/div/div[1]/div[2]/div[2]/div/div/div[2]/div[1]/div[1]/div/ul[1]/li[{}]/a/div/div[2]/strong".format(num)
            driver.find_element_by_xpath(xpath).click()
            xpath = "/html/body/div[3]/div[2]/div/div[1]/div[2]/div[1]/div[1]/h2/span[1]/strong/a"
            driver.find_element_by_xpath(xpath).click()

            driver.close()
            driver.switch_to.window(driver.window_handles[0])

            # html 정보 가져오기
            html = driver.page_source
            # html에 대해 파싱
            soup = BeautifulSoup(html, 'html.parser')

            movie_id_string = str(soup.select('meta[property="og:url"]')[0])
            movie_id = re.sub(r'[^0-9]', '', movie_id_string)# 숫자(movie id)만 남기기
            movie_id_list.append(movie_id)

        except Exception as e:
            print("Error : current_top3_movie() ")
            from IPython import embed; embed()
            driver.quit()
            return e

    driver.quit()
    return movie_id_list














if __name__ == '__main__':
    main()
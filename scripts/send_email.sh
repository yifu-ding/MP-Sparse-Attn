exp_name=$1

python helper.py \
  --smtp_server smtp.gmail.com \
  --smtp_port 465 \
  --sender_email eveedyf@gmail.com \
  --sender_password ncoilrplkdnshcqt \
  --receiver_email yifuding@qq.com  \
  --subject "exp has done!" \
  --body "$exp_name has done!"

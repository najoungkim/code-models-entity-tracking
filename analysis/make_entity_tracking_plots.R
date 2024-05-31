library(tidyverse)
library(ggplot2)

theme_set(theme_bw())
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

compute_upper_ci = function(val, n) {
  val = val / 100.0
  error = 1 - val
  const = 1.96 #95% CI
  upper = error - const * sqrt( ((error) * (1 - error)) / n)
  return(pmin(1, 1-upper) * 100)
}

compute_lower_ci = function(val, n) {
  val = val / 100.0
  error = 1 - val
  const = 1.96 #95% CI
  lower = error + const * sqrt( ((error) * (1 - error)) / n)
  return(pmax(0, 1-lower) * 100)
}

d = read.csv("~/Dropbox/Uni/RA/implicit-meaning-lms/code-models-entity-tracking/analysis/model_results.csv")

d = d %>%
  mutate(ci_upper = compute_upper_ci(accuracy, count)) %>%
  mutate(ci_lower = compute_lower_ci(accuracy, count))

d %>% ggplot(aes(x=num_ops, y=accuracy)) + geom_line() + facet_wrap(~model_name, ncol = 5)




d = d %>% mutate(model_name_short = str_replace(model_name, ".*\\(.*/(.*)\\)", "\\1"))

p = d %>% filter(model_name_short %in% c("Llama-2-7b-hf", 
                                   "CodeLlama-7b-hf", 
                                   "Llama-2-13b-hf", 
                                   "CodeLlama-13b-hf",
                                   "Llama-2-70b-hf",
                                   "CodeLlama-70b-hf", 
                                   "deepseek-llm-7b-base",
                                   "deepseek-coder-7b-base-v1.5",
                                   "gemma-7b",
                                   "codegemma-7b")) %>% 
  mutate(code = grepl("codegemma|codellama|-coder", model_name, perl = TRUE)) %>%
  mutate(model_name = str_replace(model_name, "-hf", "")) %>%
  mutate(title= str_replace(model_name_short, "(Code)?Llama-?2?-([1370]+).*$", "Llama 2 (\\2B)")) %>%
  mutate(title= str_replace(title, "deepseek.*$", "DeepSeek (7B)")) %>%
  mutate(title= str_replace(title, "(code)?gemma.*$", "Gemma (8B)")) %>%
  mutate(title = factor(title, levels = c("DeepSeek (7B)", "Gemma (8B)", "Llama 2 (7B)", "Llama 2 (13B)", "Llama 2 (70B)"))) %>%
  mutate(code = factor(code, levels=c(FALSE, TRUE), labels=c("no", "yes"))) %>%
  ggplot(aes(x=num_ops, y=accuracy, col=code)) + geom_line()  + geom_point(aes(pch=code)) + geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper), width=.1) +
  theme(legend.position = "bottom",
        legend.text = element_text(size=12)) +
  xlab("Number of operations affecting box state") +
  ylab("Accuracy") +
  scale_x_continuous(breaks=0:7) +
  scale_linetype_discrete(limits=c(TRUE, FALSE)) +
  guides(col=guide_legend(nrow=1, byrow=TRUE, title = "Additional code training"),
         pch=guide_legend(nrow=1, byrow=TRUE, title = "Additional code training")) +
  facet_wrap(~title, ncol=5) +
  geom_line(col="black", lty=2, data= d %>% filter(model_name_short=="random"))
p

ggsave("base-vs-code.pdf", plot=p, width=20, height=6, units="cm")


p = d %>% filter(model_name_short %in% c("Llama-2-7b-hf", 
                                         "CodeLlama-7b-hf", 
                                         "Llama-2-13b-hf", 
                                         "CodeLlama-13b-hf",
                                         "Llama-2-70b-hf",
                                         "CodeLlama-70b-hf", 
                                         "deepseek-llm-7b-base",
                                         "deepseek-coder-7b-base-v1.5",
                                         "deepseek-coder-7b-instruct-v1.5",
                                         "deepseek-llm-7b-chat",
                                         "Llama-2-7b-chat-hf",
                                         "Llama-2-13b-chat-hf",
                                         "Llama-2-70b-chat-hf",
                                         "CodeLlama-7b-Instruct-hf",
                                         "CodeLlama-70b-Instruct-hf",
                                         "CodeLlama-13b-Instruct-hf",
                                         "codegemma-7b",
                                         "codegemma-7b-it",
                                         "gemma-7b",
                                         "gemma-7b-it")) %>% 
  mutate(code = grepl("codegemma|codellama|-coder", model_name, perl = TRUE)) %>%
  mutate(model_name = str_replace(model_name, "-hf", "")) %>%
  mutate(title= str_replace(model_name_short, "(Code)?Llama-?2?-([1370]+).*$", "Llama 2 (\\2B)")) %>%
  mutate(title= str_replace(title, "deepseek.*$", "DeepSeek (7B)")) %>%
  mutate(title= str_replace(title, "(code)?gemma.*$", "Gemma (8B)")) %>%
  mutate(title = factor(title, levels = c("DeepSeek (7B)", "Gemma (8B)", "Llama 2 (7B)", "Llama 2 (13B)", "Llama 2 (70B)"))) %>%
  mutate(chat = grepl("Instruct|instruct|chat|-it", model_name_short, perl=TRUE)) %>%
  mutate(chat = factor(chat, levels=c(FALSE, TRUE), labels=c("base", "instruct/chat"))) %>%
  mutate(code = factor(code, levels=c(FALSE, TRUE), labels=c("no add. code training", "add. code training"))) %>%
  ggplot(aes(x=num_ops, y=accuracy, col=chat)) + geom_line()  + geom_point(aes(pch=chat)) + geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper), width=.1) +
  theme(legend.position = "bottom",
        legend.text = element_text(size=12)) +
  xlab("Number of operations affecting box state") +
  ylab("Accuracy") +
  scale_x_continuous(breaks=0:7) +
  scale_linetype_manual(values=c("FALSE"=2, "TRUE"=1)) +
  guides(col=guide_legend(nrow=1, byrow=TRUE, title = ""),
         pch=guide_legend(nrow=1, byrow=TRUE, title = "")) + 
    facet_grid(code~title) +
  scale_color_brewer(palette="Dark2", direction = -1) +
  geom_line(col="black", group="a", alpha=0.8, lty=2, data= d %>% filter(model_name_short=="random"))


p
ggsave("instruction-tuning.pdf", plot=p, width=20, height=11, units="cm")


#math_data.float

math_data.mistral = d %>% filter(model_name_short %in% c("Mistral-7B-v0.1", 
                                                         "OpenMath-Mistral-7B-v0.1-hf")) %>%
                                   mutate(math=grepl("OpenMath", model_name_short)) %>%
                                   mutate(comparison="OpenMath-Mistral (7B)")

math_data.deepseek = d %>% filter(model_name_short %in% c("deepseek-coder-7b-base-v1.5", 
                                                         "deepseek-math-7b-base")) %>%
  mutate(math=grepl("-math-", model_name_short)) %>%
  mutate(comparison="DeepSeek-Math (7B)")

math_data.llemma7b = d %>% filter(model_name_short %in% c("CodeLlama-7b-hf", 
                                                           "llemma_7b")) %>%
  mutate(math=grepl("llemma", model_name_short)) %>%
  mutate(comparison="Llemma (7B)")


math_data.llemma34b = d %>% filter(model_name_short %in% c("CodeLlama-34b-hf", 
                                                          "llemma_34b")) %>%
  mutate(math=grepl("llemma", model_name_short)) %>%
  mutate(comparison="Llemma (34B)")

math_data.float = d %>% filter(model_name_short %in% c("llama-7b", 
                                                           "float-7b")) %>%
  mutate(math=grepl("float", model_name_short)) %>%
  mutate(comparison="FLoat (7B)")

math_data = rbind(math_data.mistral, math_data.deepseek, math_data.llemma7b, math_data.llemma34b, math_data.float)

math_data = math_data %>%
  mutate(math = factor(math, levels=c(FALSE, TRUE), c("no", "yes")))

p = math_data %>%
  mutate(model_name = str_replace(model_name, "-hf", "")) %>%
  mutate(comparison = factor(comparison, levels = c("FLoat (7B)", "OpenMath-Mistral (7B)", "DeepSeek-Math (7B)", "Llemma (7B)", "Llemma (34B)"))) %>%
  ggplot(aes(x=num_ops, y=accuracy, col=math)) + geom_line()  + geom_point(aes(pch=math)) + geom_errorbar(aes(ymin=ci_lower, ymax=ci_upper), width=.1) +
  theme(legend.position = "bottom") +
  xlab("Number of operations affecting box state") +
  ylab("Accuracy") +
  scale_x_continuous(breaks=0:7) +
  scale_linetype_discrete(limits=c(TRUE, FALSE)) +
  guides(col=guide_legend(nrow=1, byrow=TRUE, title = "Additional math training"), pch=guide_legend(nrow=1, byrow=TRUE, title = "Additional math training")) +
  facet_wrap(~comparison, ncol=5) +
  geom_line(col="black", lty=2, data= d %>% filter(model_name_short=="random"))


p
ggsave("math-tuning.pdf", plot=p, width=20, height=6, units="cm")


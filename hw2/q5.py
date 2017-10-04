background, salt, pepper = 50, 100, 0
mask = [-1, 2, -1]

pr_background_salt_salt = 0.3*0.3
val_background_salt_salt = [background*mask[0], salt*mask[1], salt*mask[2]]
print "pr_background_salt_salt: ", pr_background_salt_salt, "val_background_salt_salt", val_background_salt_salt

pr_background_pepper_pepper = 0.7*0.7
val_background_pepper_pepper = [background*mask[0], pepper*mask[1], pepper*mask[2]]
print "pr_background_pepper_pepper: ", pr_background_pepper_pepper, "val_background_pepper_pepper", val_background_pepper_pepper

pr_background_salt_pepper = 0.3*0.7
val_background_salt_pepper = [background*mask[0], salt*mask[1], pepper*mask[2]]
print "pr_background_salt_pepper: ", pr_background_salt_pepper, "val_background_salt_pepper", val_background_salt_pepper

pr_background_pepper_salt = 0.7*0.3
val_background_pepper_salt = [background*mask[0], pepper*mask[1], salt*mask[2]]
print "pr_background_pepper_salt: ", pr_background_pepper_salt, "val_background_pepper_salt", val_background_pepper_salt

